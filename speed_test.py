import time
import statistics
from transformers import BertForSequenceClassification, AutoTokenizer, TextClassificationPipeline

MODEL_PATH = "./model"
CHUNK_SIZE = 200
CHUNK_STEP = 150
THRESHOLD  = 0.5

print("모델 로딩 중...")
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
pipe = TextClassificationPipeline(
    model=model, tokenizer=tokenizer,
    device=-1, top_k=None,
    function_to_apply='sigmoid',
    truncation=True, max_length=512
)
print("모델 로딩 완료!\n")

def split_chunks(text):
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_STEP
    return chunks

def run(text):
    chunks = split_chunks(text)
    label_max = {}
    for chunk in chunks:
        for item in pipe(chunk)[0]:
            label, score = item['label'], item['score']
            if label not in label_max or score > label_max[label]:
                label_max[label] = score
    detected = [l for l, s in label_max.items() if s >= THRESHOLD and l != 'clean']
    return len(chunks), bool(detected)

# ─── 테스트 케이스 ────────────────────────────────────────────
cases = {
    "짧은 댓글 (20자)":
        "오늘 날씨 정말 좋네요!",

    "일반 댓글 (100자)":
        "이 API 정말 유용하네요. 개발하면서 항상 찾던 기능인데 이렇게 잘 정리돼 있으니까 너무 좋습니다. 앞으로도 자주 이용할게요.",

    "혐오 포함 댓글 (50자)":
        "야이 XX야 이딴 걸 올려놓으면 어떡하냐 진짜 XX같은 놈아",

    "중간 게시글 (500자)":
        "안녕하세요. GetAPI를 사용하면서 느낀 점을 공유합니다. " * 10,

    "긴 게시글 (1000자)":
        "개발자라면 누구나 공개 API가 필요한 순간이 있습니다. 날씨, 번역, 지도, 금융 데이터 등 다양한 API를 한 곳에서 찾고 리뷰까지 볼 수 있다면 정말 편리하겠죠. " * 8,

    "매우 긴 게시글 (3000자)":
        "GetAPI 플랫폼은 국내외 다양한 공개 API를 카테고리별로 정리하고 사용자 리뷰와 평점을 제공하는 서비스입니다. " * 30,

    "혐오 문장이 뒤에 숨은 긴 글 (500자 + 혐오)":
        ("오늘은 맑은 하늘 아래 좋은 하루를 보냈습니다. 커피도 맛있고 날씨도 좋고. " * 7)
        + " 야이 XX놈아 진짜 꺼져버려.",
}

REPEAT = 3
print(f"{'케이스':<30} {'글자수':>6} {'청크수':>5} {'차단':>5} {'평균시간':>10} {'최소':>8} {'최대':>8}")
print("─" * 80)

for name, text in cases.items():
    times = []
    chunks_n = 0
    censored = False
    for _ in range(REPEAT):
        t0 = time.time()
        chunks_n, censored = run(text)
        times.append(time.time() - t0)

    avg = statistics.mean(times)
    mn  = min(times)
    mx  = max(times)
    print(f"{name:<30} {len(text):>6} {chunks_n:>5} {'YES' if censored else 'NO':>5} "
          f"{avg:>9.3f}s {mn:>7.3f}s {mx:>7.3f}s")

print("\n완료!")
