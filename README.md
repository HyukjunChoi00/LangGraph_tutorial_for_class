# LangGraph 뉴스 검색 워크플로우

키워드 추출 → 쿼리 확장 → 뉴스 검색 → 최종 답변의 워크플로우를 구현한 LangGraph 애플리케이션입니다.

## 워크플로우 구조

1. **키워드 추출** (`extract_keyword`): 사용자 질문에서 핵심 검색 키워드를 추출
2. **쿼리 확장** (`query_expansion`): 추출된 키워드를 관련 산업/기술 키워드로 확장
3. **뉴스 검색** (`search_news`): 확장된 키워드들로 Econotimes에서 뉴스 기사 검색
4. **최종 답변** (`generate_response`): 검색 결과를 바탕으로 마크다운 리포트 스타일 답변 생성

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
playwright install
```

### 2. 환경 변수 설정
Google Gemini API 키가 필요합니다:
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

### 3. 실행
```bash
python langgraph_news_workflow.py
```

## 주요 기능

- **비동기 뉴스 스크래핑**: Playwright를 사용한 Econotimes 뉴스 검색
- **지능형 키워드 확장**: Gemini 모델을 활용한 관련 키워드 생성
- **마크다운 리포트**: 구조화된 답변 생성
- **상태 관리**: LangGraph의 StateGraph를 활용한 워크플로우 상태 추적

## 사용 예시

```python
# 단일 쿼리 실행
result = await run_workflow("테슬라 주식 최신 동향 알려줘")

# 여러 쿼리 테스트 (main 함수에서 실행됨)
test_queries = [
    "테슬라 주식 최신 동향 알려줘",
    "NVIDIA AI 칩 관련 소식이 궁금해", 
    "애플 아이폰 신제품 출시 소식"
]
```

## State 구조

```python
class State(TypedDict):
    query: str                    # 원본 사용자 질문
    messages: Annotated[list, add_messages]  # 대화 메시지 리스트
    extracted_keyword: str        # 추출된 핵심 키워드
    expanded_keywords: list[str]  # 확장된 키워드 리스트
    search_results: str          # 뉴스 검색 결과
    answer: str                  # 최종 답변
```

## 참고사항

- Playwright 브라우저가 headless 모드로 실행됩니다
- 각 키워드당 최대 2개의 기사를 검색합니다
- 기사 본문은 800자로 제한됩니다
- 에러 처리 및 재시도 로직이 포함되어 있습니다