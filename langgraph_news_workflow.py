import json
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from playwright.async_api import async_playwright


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class State(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    extracted_keyword: str
    expanded_keywords: list[str]
    search_results: str
    answer: str

def build_keyword_extraction_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 사용자의 질문에서 핵심 검색 키워드를 추출하는 Assistant이다."),
         ("user", "다음 사용자의 질문에서 검색에 사용할 핵심 키워드만 추출하라. 반드시 영문 키워드로 제공하라.\n"
                  "예시:\n"
                  "- 'NVIDIA 주식 동향 조사좀 해줘' → 'NVIDIA'\n"
                  "- '테슬라 주가 전망이 어때?' → 'Tesla'\n"
                  "- '애플 아이폰 최신 소식 알려줘' → 'Apple'\n"
                  "- 'AI 기술 발전 현황' → 'AI'\n"
                  "질문: {query}\n\n"
                  "다음 JSON 형식으로만 응답해줘:\n"
                  '{{"extracted_keyword": "추출된_키워드"}}\n'
                  "핵심 키워드:")]
    )
    
    return prompt_template | llm


def build_query_expansion_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 키워드형 Query Expansion Assistant이다. 주어진 키워드와 관련된 산업, 기술, 시장 키워드로 확장해야 한다."),
         ("user", "다음 키워드를 확장하라:\n"
                  "원본 키워드: {query}\n"
                  "확장할 개수: {n}개\n\n"
                  "확장 규칙:\n"
                  "- Tesla → electric vehicle, EV, battery\n"
                  "- NVIDIA → AI, GPU, semiconductor\n"
                  "- Apple → iPhone, technology, consumer electronics\n"
                  "- Microsoft → cloud computing, software, Azure\n\n"
                  "'{query}' 키워드에 대해 관련 산업/기술 키워드 {n}개를 영문으로 생성하라.\n\n"
                  "반드시 다음 JSON 형식으로만 응답하라 (다른 텍스트 포함 금지):\n"
                  '{{"expanded_search_query_list": ["키워드1", "키워드2"]}}\n')]
    )
    
    return prompt_template | llm


def generate_response(state):
    context = state.get("search_results", "")
    
    curr_human_turn = HumanMessage(content=f"질문: {state['query']}\n"
                            f"검색 결과:\n```\n{context}```"
                             "\n---\n"
                             "응답은 markdown을 이용해 리포트 스타일로 한국어로 응답해라. "
                             "사용자의 질문의 의도에 맞는 정답 부분을 강조해라.")
    messages = state["messages"] + [curr_human_turn]
    response = llm.invoke(messages)

    return {"messages": [*messages, response],
            "answer": response.content}


def parse_json_response(response) -> dict:
    """JSON 응답을 파싱하는 간단한 함수"""

    # AIMessage 객체에서 content 추출
    content = str(getattr(response, 'content', response))

    # JSON 블록 찾기
    if '```json' in content:
        start = content.find('```json') + 7
        end = content.find('```', start)
        json_str = content[start:end].strip()
    elif '{' in content and '}' in content:
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
    else:
        return {"extracted_keyword": "", "expanded_search_query_list": []}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"extracted_keyword": "", "expanded_search_query_list": []}


def extract_keyword(state):
    """핵심 키워드 추출 노드"""
    print(f"🔍 키워드 추출 중: {state['query']}")
    
    keyword_extraction_chain = build_keyword_extraction_chain()
    original_query = state["query"]
    response = keyword_extraction_chain.invoke({"query": original_query})
    parsed_response = parse_json_response(response)
    
    extracted_keyword = parsed_response.get("extracted_keyword", "")
    print(f"✅ 추출된 키워드: {extracted_keyword}")
    
    return {"extracted_keyword": extracted_keyword}


def query_expansion(state):
    """쿼리 확장 노드"""
    print(f"🔄 쿼리 확장 중: {state['extracted_keyword']}")
    
    query_expansion_chain = build_query_expansion_chain()
    extracted_keyword = state["extracted_keyword"]
    response = query_expansion_chain.invoke({"query": extracted_keyword, "n": 2})
    parsed_response = parse_json_response(response)
    
    expanded_keywords = parsed_response.get("expanded_search_query_list", [])
    # 원본 키워드도 포함
    all_keywords = [extracted_keyword] + expanded_keywords
    print(f"✅ 확장된 키워드: {all_keywords}")
    
    return {"expanded_keywords": all_keywords}


async def search_news(state):
    """뉴스 검색 노드"""
    print(f"📰 뉴스 검색 중...")
    
    expanded_keywords = state.get("expanded_keywords", [])
    all_results = []
    
    # 각 키워드로 검색 수행
    for keyword in expanded_keywords[:2]:  # 최대 2개 키워드만 사용
        if keyword:
            print(f"   검색 키워드: {keyword}")
            try:
                result = await scrape_articles_with_content(keyword, max_articles=2)
                if result and "오류" not in result and "실패" not in result:
                    all_results.append(f"=== {keyword} 검색 결과 ===\n{result}\n")
            except Exception as e:
                print(f"   ❌ {keyword} 검색 실패: {e}")
                continue
    
    search_results = "\n".join(all_results) if all_results else "검색 결과가 없습니다."
    print(f"✅ 검색 완료: {len(all_results)}개 결과")
    
    return {"search_results": search_results}


# 비동기 스크래핑 도구 정의
@tool
async def scrape_articles_with_content(query: str, max_articles: int = 3) -> str:
    """
    Econotimes에서 관련 기사 제목, URL, 본문을 스크랩하는 비동기 함수
    사용자 특정 종목에 대한 동향 분석 등을 요청할 때 이 도구를 이용해 뉴스 기사를 검색합니다.
    
    
    Args:
        query: 검색할 키워드 (예: tesla, apple, bitcoin 등)
        max_articles: 추출할 최대 기사 수 (기본값: 3)
    
    Returns:
        기사 정보가 포함된 JSON 형태의 문자열
    """
    print(f"🚀 Econotimes에서 '{query}' 검색 중...")
    
    output_list = []
    
    try:
        async with async_playwright() as p:
            # 브라우저 실행
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # 검색 페이지로 이동
            search_url = f"https://econotimes.com/search?v={query}&search="
            await page.goto(search_url)
            await asyncio.sleep(2)
            
            # XPath를 사용해서 모든 기사 제목 요소 찾기
            general_xpath = '//*[@id="archivePage"]/div/div[2]/div/p[1]/a'
            elements = await page.locator(f"xpath={general_xpath}").all()
            
            
            if not elements:
                await browser.close()
                return f"'{query}'에 대한 기사를 찾을 수 없습니다."
            
            # 지정된 개수만큼 기사 처리
            for i, element in enumerate(elements[:max_articles], 1):
                try:
                    # 기사 제목과 링크 추출
                    title = await element.text_content()
                    href = await element.get_attribute('href')
                    
                    if title and href:
                        title = title.strip()
                        full_url = f"https://econotimes.com{href}" if href.startswith('/') else href
                        
                        print(f"{i}. {title}")
                        
                        # 새 탭에서 기사 본문 추출
                        article_page = await browser.new_page()
                        try:
                            await article_page.goto(full_url)
                            await asyncio.sleep(2)
                            
                            # 본문 추출
                            article_xpath = '//*[@id="view"]/div[2]/div[3]/article'
                            article_content = await article_page.locator(f"xpath={article_xpath}").text_content()
                            
                            if article_content:
                                article_content = article_content.strip()
                                # 본문이 너무 길면 앞부분만
                                content_preview = article_content[:800] + "..." if len(article_content) > 800 else article_content
                            else:
                                content_preview = "본문을 추출할 수 없습니다."
                            
                            output_list.append({
                                'number': i,
                                'title': title,
                                'url': full_url,
                                'content': content_preview
                            })
                            
                            
                            
                        except Exception as e:
                            print(f"   ❌ 본문 추출 실패: {e}")
                            output_list.append({
                                'number': i,
                                'title': title,
                                'url': full_url,
                                'content': '본문 추출 실패'
                            })
                        finally:
                            await article_page.close()
                
                except Exception as e:
                    print(f"{i}. ❌ 기사 처리 실패: {e}")
                    continue
            
            await browser.close()
            
            if output_list:

                # JSON 형태로 반환
                import json
                return json.dumps(output_list, ensure_ascii=False, indent=2)
            else:
                return f"'{query}' 기사 추출에 실패했습니다."
                
    except Exception as e:
        return f"스크래핑 중 오류 발생: {str(e)}"


# 워크플로우 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("extract_keyword", extract_keyword)
workflow.add_node("query_expansion", query_expansion)
workflow.add_node("search_news", search_news)
workflow.add_node("generate_response", generate_response)

# 엣지 연결
workflow.add_edge(START, "extract_keyword")
workflow.add_edge("extract_keyword", "query_expansion")
workflow.add_edge("query_expansion", "search_news")
workflow.add_edge("search_news", "generate_response")
workflow.add_edge("generate_response", END)

# 그래프 컴파일
graph = workflow.compile()


async def run_workflow(query: str):
    """워크플로우 실행 함수"""
    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "extracted_keyword": "",
        "expanded_keywords": [],
        "search_results": "",
        "answer": ""
    }
    
    print(f"🚀 워크플로우 시작: {query}")
    print("="*50)
    
    # 비동기로 그래프 실행
    final_state = None
    async for state in graph.astream(initial_state):
        final_state = state
    
    print("="*50)
    print("✅ 워크플로우 완료!")
    
    return final_state


# 실행 예시
async def main():
    # 테스트 쿼리
    test_queries = [
        "테슬라 주식 최신 동향 알려줘",
        "NVIDIA AI 칩 관련 소식이 궁금해",
        "애플 아이폰 신제품 출시 소식"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"테스트 쿼리: {query}")
        print(f"{'='*60}")
        
        try:
            result = await run_workflow(query)
            if result:
                # 최종 상태에서 답변 추출
                for node_name, node_state in result.items():
                    if "answer" in node_state and node_state["answer"]:
                        print(f"\n📋 최종 답변:")
                        print(node_state["answer"])
                        break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # 비동기 실행
    asyncio.run(main())