# GetAPI Censored AI

GetAPI 플랫폼에서 사용하는 한국어 혐오표현 감지 AI 서버입니다.  
[smilegate-ai/kor_unsmile](https://huggingface.co/smilegate-ai/kor_unsmile) BERT 모델을 기반으로, API 이름·설명·게시글 등의 텍스트를 자동으로 검열합니다.

---

## 기능

- **한국어 혐오표현 다중 레이블 분류** (9개 카테고리 + clean)
- **긴 텍스트 청크 분할 처리** — 최대 5,000자, 경계 겹침 방식으로 누락 방지
- **비동기 추론** — FastAPI + `asyncio.to_thread`로 블로킹 없이 처리
- **일별 로그 파일** 자동 생성 (`logs/YYYYMMDD.log`)

### 감지 카테고리

| 레이블 | 설명 |
|---|---|
| 여성/가족 | 여성 및 가족 대상 혐오 |
| 남성 | 남성 대상 혐오 |
| 성소수자 | 성소수자 대상 혐오 |
| 인종/국적 | 인종·국적 대상 혐오 |
| 연령 | 연령 대상 혐오 |
| 지역 | 지역 대상 혐오 |
| 종교 | 종교 대상 혐오 |
| 기타 혐오 | 기타 혐오표현 |
| 악플/욕설 | 욕설·악성 댓글 |
| clean | 정상 텍스트 |

---

## 요구사항

- Python 3.10 이상
- git-lfs (모델 파일 다운로드에 필요)

---

## 설치 및 실행

### 1. git-lfs 설치

모델 파일(`model.safetensors`, 416MB)이 Git LFS로 관리됩니다.  
클론 전에 반드시 git-lfs를 먼저 설치해야 합니다.

```bash
# Ubuntu / Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Arch Linux
sudo pacman -S git-lfs
```

### 2. 레포 클론

```bash
git lfs install
git clone https://github.com/devlib-itsw/GetAPI_Censored_Ai.git
cd GetAPI_Censored_Ai
```

> git-lfs가 설치된 상태에서 클론하면 모델 파일까지 자동으로 다운로드됩니다.  
> 이미 클론한 경우 `git lfs pull`을 실행하면 모델 파일을 받을 수 있습니다.

### 3. 가상환경 생성 및 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. 실행

```bash
python main.py
```

서버가 `http://127.0.0.1:8888`에서 시작됩니다.

---

## 디렉터리 구조

```
.
├── main.py                      # FastAPI 서버 메인
├── learn.py                     # 모델 다운로드 스크립트
├── speed_test.py                # 추론 속도 테스트
├── requirements.txt             # 의존성 목록
├── model/                       # BERT 모델 파일 (Git LFS)
│   ├── config.json
│   ├── model.safetensors        # 416MB — Git LFS로 관리
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
└── logs/                        # 실행 시 자동 생성
    └── YYYYMMDD.log
```

---

## API 사용법

### `GET /health` — 서버 상태 확인

```http
GET http://127.0.0.1:8888/health
```

**응답**
```json
{
  "status": "ok",
  "model": "./model"
}
```

---

### `POST /check` — 텍스트 검열

```http
POST http://127.0.0.1:8888/check
Content-Type: application/json
```

**요청 바디**
```json
{
  "text": "검열할 텍스트를 입력합니다."
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `text` | string | 검사할 텍스트 (1자 이상, 5,000자 이하) |

**응답**
```json
{
  "is_censored": false,
  "detected_labels": [],
  "scores": {
    "여성/가족": 0.0021,
    "남성": 0.0015,
    "성소수자": 0.0008,
    "인종/국적": 0.0012,
    "연령": 0.0009,
    "지역": 0.0006,
    "종교": 0.0007,
    "기타 혐오": 0.0031,
    "악플/욕설": 0.0018,
    "clean": 0.9980
  },
  "chunk_count": 1
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `is_censored` | boolean | 혐오표현 감지 여부 |
| `detected_labels` | string[] | 임계값 초과 카테고리 목록 |
| `scores` | object | 카테고리별 점수 (0.0 ~ 1.0) |
| `chunk_count` | integer | 분할된 청크 수 |

**오류 응답**

| 상태 코드 | 원인 |
|---|---|
| `422` | 텍스트가 비어있거나 5,000자 초과 |
| `500` | 모델 추론 실패 |

---

## 설정값

`main.py` 상단에서 조정할 수 있습니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `THRESHOLD` | `0.7` | 혐오 판정 임계값 (0.0~1.0) |
| `MAX_LENGTH` | `5000` | 최대 입력 글자 수 |
| `CHUNK_SIZE` | `200` | 청크당 글자 수 |
| `CHUNK_STEP` | `150` | 청크 이동 간격 (50자 겹침) |

---

## GetAPI 연동

GetAPI Spring Boot 서버의 `application.properties`에 아래 설정을 추가합니다.

```properties
ai.server.url=http://127.0.0.1:8888
```

API 등록·수정 시 GetAPI 서버가 이 주소로 `/check` 요청을 보내 자동 검열합니다.

---

## 라이선스

기반 모델 [smilegate-ai/kor_unsmile](https://github.com/smilegate-ai/korean_unsmile_dataset)은 Apache 2.0 라이선스로 공개되어 있습니다.
