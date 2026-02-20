# Web Chat UI (Vanilla JS + Tailwind)

프론트엔드는 `/chat` API를 호출하는 멀티턴 진로상담 챗봇 UI입니다.

## 실행

FastAPI에서 정적 마운트:

```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="web", html=True), name="web")
```

서버 실행 후 `http://localhost:8000` 접속.

## 기능

- 학생 진로상담 챗 대화창
- 멀티턴 history 전달
- 도메인 필터 입력(학교급/추천직업 카테고리)
- 빠른 질문 템플릿
- 답변별 근거(citations) 노출
