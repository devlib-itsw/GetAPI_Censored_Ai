from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from pydantic import BaseModel, field_validator
import uvicorn
import asyncio
import logging
import os
from datetime import datetime

# ─── 로깅 설정 ────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─── 설정 ─────────────────────────────────────────────────────
MODEL_PATH  = "./model"
THRESHOLD   = 0.7   # 이 스코어 이상이면 혐오 판정
MAX_LENGTH  = 5000  # 최대 입력 글자 수
CHUNK_SIZE  = 200   # 청크당 글자 수 (한국어 기준 약 400~500토큰)
CHUNK_STEP  = 150   # 청크 이동 간격 (50자 겹침으로 경계 누락 방지)

# ─── 앱 초기화 ────────────────────────────────────────────────
app = FastAPI(title="AI Filter Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── 모델 로딩 ────────────────────────────────────────────────
logger.info("모델 로딩 중...")
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=-1,
    top_k=None,
    function_to_apply='sigmoid',
    truncation=True,
    max_length=512
)
logger.info("모델 로딩 완료!")

# ─── 청크 분할 ────────────────────────────────────────────────
def split_chunks(text: str) -> list[str]:
    """텍스트를 CHUNK_SIZE 글자씩, CHUNK_STEP 간격으로 분할 (경계 겹침 포함)"""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_STEP
    return chunks

def analyze_chunks(chunks: list[str]) -> dict:
    """
    모든 청크를 검사하여 카테고리별 최대 스코어 반환.
    하나의 청크라도 혐오가 감지되면 전체 차단.
    """
    label_max_scores: dict[str, float] = {}

    for chunk in chunks:
        results = pipe(chunk)[0]
        for item in results:
            label = item['label']
            score = item['score']
            if label not in label_max_scores or score > label_max_scores[label]:
                label_max_scores[label] = score

    return label_max_scores

# ─── 요청/응답 모델 ───────────────────────────────────────────
class CheckRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("텍스트가 비어있습니다.")
        if len(v) > MAX_LENGTH:
            raise ValueError(f"텍스트는 {MAX_LENGTH}자 이하여야 합니다. (현재: {len(v)}자)")
        return v.strip()

class CheckResponse(BaseModel):
    is_censored: bool
    detected_labels: list[str]
    scores: dict[str, float]
    chunk_count: int

# ─── 엔드포인트 ───────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/check", response_model=CheckResponse)
async def check_text(req: CheckRequest):
    try:
        chunks = split_chunks(req.text)

        scores = await asyncio.to_thread(analyze_chunks, chunks)

        detected_labels = [
            label for label, score in scores.items()
            if score >= THRESHOLD and label != 'clean'
        ]
        is_censored = len(detected_labels) > 0

        logger.info(
            f"[{'BLOCK' if is_censored else 'PASS '}] "
            f"chunks={len(chunks)} labels={detected_labels} | "
            f"text={req.text[:50]}..."
        )

        return CheckResponse(
            is_censored=is_censored,
            detected_labels=detected_labels,
            scores={k: round(v, 4) for k, v in scores.items()},
            chunk_count=len(chunks)
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"추론 실패: {e}")
        raise HTTPException(status_code=500, detail="분석 중 오류가 발생했습니다.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)