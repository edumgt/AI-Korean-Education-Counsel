# AI 기반 학생 진로탐색 상담 시스템 (Chatbot RAG)

이 저장소는 `DATA_ROOT`의 학생 기초정보/상담기록/전문가 라벨링 데이터를 기반으로, **백엔드(FastAPI) + 프론트엔드(Web) 모두 챗봇 중심**으로 동작하는 진로상담 시스템입니다.

---

## 1. 시스템 개요

- **목표**: 학생의 흥미·강점·상담 맥락을 반영한 근거 기반 진로탐색 상담 제공
- **방식**: RAG(Retrieval-Augmented Generation)
  1. `DATA_ROOT` 원천/라벨링 데이터를 정규화
  2. 임베딩 후 Qdrant 벡터 검색 인덱스 구축
  3. `/chat` API로 문맥+근거 기반 답변 생성
  4. 프론트 챗 UI에서 멀티턴 상담 수행

---

## 2. 기술 스택 (상세)

### Backend
- **FastAPI**: REST API 서버 (`/chat`, `/ask`, `/healthz`)
- **Pydantic**: 요청/응답 스키마 검증
- **Sentence-Transformers (intfloat/multilingual-e5-small)**: 한글 질의/문서 임베딩
- **Qdrant**: 벡터 검색 DB (전체 + 도메인별 컬렉션)
- **LLM Provider (선택형)**
  - `none`: LLM 없이 근거 요약 기반 폴백 응답
  - `ollama`: 로컬 모델 연동
  - `openai`: OpenAI Chat Completions 연동
- **Python Requests**: Ollama HTTP 호출

### Data Pipeline
- `scripts/normalize.py`
  - `학생기초정보`, `상담기록`, `전문가_라벨링` JSON을 통합 파싱
  - `documents.jsonl`(검색 문서), `qas.jsonl`(평가용 QA) 생성
- `scripts/index_qdrant.py`
  - 문서 청킹(char 기반)
  - 임베딩 생성
  - `career_all` + `career_{domain}` 컬렉션 업서트

### Frontend
- **Vanilla JavaScript + TailwindCSS(CDN)**
- 챗 UI 기능
  - 멀티턴 대화 이력 전송
  - 도메인(학교급/카테고리) 필터
  - 빠른 질문 템플릿
  - 답변별 근거(citations) 표시

### Infra / Runtime
- **Docker Compose**: Qdrant 실행
- **Uvicorn**: FastAPI ASGI 서버 구동
- **python-dotenv**: `.env` 환경변수 로딩

---

## 3. 데이터 구조 가정

`DATA_ROOT` 예시:

```text
DATA_ROOT/
  01.원천데이터/
    01. 학교급/
      01. 초등/
      02. 중등/
      03. 고등/
  02.라벨링데이터/
    01. 학교급/
      01. 초등/
      02. 중등/
      03. 고등/
    02. 추천직업 카테고리/
      01. 기술계열/
      02. 서비스계열/
      03. 생산계열/
      04. 사무계열/
```

---

## 4. 실행 방법

### 4-1. 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 4-2. Qdrant 실행

```bash
docker compose up -d qdrant
```

### 4-3. 데이터 정규화

```bash
python3 scripts/normalize.py --data-root DATA_ROOT --out-dir data
```

### 4-4. 임베딩/인덱싱

```bash
python3 scripts/index_qdrant.py --docs data/documents.jsonl
```

### 4-5. API + 웹 실행

```bash
uvicorn api.main:app --reload --port 8000
```

브라우저: `http://localhost:8000`

---

## 5. API 요약

### POST `/chat`
멀티턴 상담 챗 API

요청 예시:
```json
{
  "message": "이 학생에게 맞는 진로를 제안해줘",
  "domain": "01. 학교급 / 03. 고등",
  "top_k": 5,
  "history": [
    {"role": "user", "content": "학생은 만들기 활동을 좋아해"}
  ]
}
```

응답: `answer`, `citations[]`, `used_collection`

### POST `/ask`
단일 질의 호환 API (`query` 기반)

### GET `/healthz`
헬스체크 + 설정 확인

---

## 6. LLM 환경변수

`.env` 예시:

```env
QDRANT_URL=http://localhost:6333
EMBED_MODEL=intfloat/multilingual-e5-small

# none | ollama | openai
LLM_PROVIDER=none

# ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

---

## 7. 운영 시 권장사항

- 상담 데이터는 민감정보 가능성이 있어 비식별화/접근통제가 필요합니다.
- LLM 생성 답변은 보조 의견이며, 실제 진학/진로 결정은 교사·상담사와 함께 검토해야 합니다.
- 학교급/직업카테고리별 분리 인덱스로 검색 품질을 점검하세요.
