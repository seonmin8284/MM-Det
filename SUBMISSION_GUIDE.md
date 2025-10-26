# AI Factory 제출 가이드

이 문서는 MM-Det 모델을 AI Factory 경진대회에 제출하는 방법을 안내합니다.

## 📋 목차
1. [제출 규칙](#제출-규칙)
2. [프로젝트 구조](#프로젝트-구조)
3. [제출 전 체크리스트](#제출-전-체크리스트)
4. [제출 방법](#제출-방법)
5. [주의사항](#주의사항)
6. [문제 해결](#문제-해결)

---

## 📌 제출 규칙

### 평가 방법
- **자동 채점**: 비공개 테스트 데이터셋으로 자동 채점
- **평가 지표**: Macro F1-score
- **추론 시간 제한**: 최대 3시간

### submission.csv 형식
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| filename | 문자열 | 파일명 (확장자 포함) |
| label | 정수 | 0 (real) 또는 1 (fake) |

**예시:**
```csv
filename,label
image001.jpg,0
video001.mp4,1
image002.png,0
```

### 평가 데이터 경로
```
./data/
    ├── {이미지 데이터1}.jpg
    ├── {이미지 데이터2}.png
    ├── {동영상 데이터1}.mp4
    ├── {동영상 데이터2}.mp4
    ...
```

---

## 📁 프로젝트 구조

제출시 다음과 같은 구조로 압축하여 제출합니다:

```
📁 프로젝트 루트/
├── 📃 task.ipynb                     # [필수] 제출용 노트북
├── 📁 model/                          # 모델 폴더
│   └── 📁 deep-fake-detector-v2-model/
│       ├── README.md
│       └── (추론 결과 저장 폴더)
├── 📁 weights/                        # [필수] 사전 학습된 가중치
│   ├── 📁 MM-Det/
│   │   └── current_model.pth
│   └── 📁 ViT/
│       └── vit_base_r50_s16_224.orig_in21k
├── 📁 LLaVA/                          # [필수] LLaVA 모델
│   ├── llava/
│   └── pyproject.toml
├── 📁 models/                         # [필수] 모델 아키텍처
│   ├── __init__.py
│   └── (기타 모델 파일들)
├── 📁 dataset/                        # [필수] 데이터셋 로더
├── 📁 utils/                          # [필수] 유틸리티
├── 📁 options/                        # 옵션 설정
└── builder.py                         # 모델 빌더

⚠️ 제외할 폴더/파일:
├── .git/                              # [제외] Git 폴더
├── .venv/                             # [제외] 가상환경
├── data/                              # [제외] 로컬 데이터
└── __pycache__/                       # [제외] 캐시 파일
```

---

## ✅ 제출 전 체크리스트

### 1. 필수 파일 확인
- [ ] `task.ipynb` 파일 존재
- [ ] `weights/MM-Det/current_model.pth` 파일 존재
- [ ] `weights/ViT/vit_base_r50_s16_224.orig_in21k` 파일 존재
- [ ] `LLaVA/` 폴더 존재
- [ ] `models/` 폴더 존재
- [ ] `dataset/` 폴더 존재
- [ ] `utils/` 폴더 존재

### 2. task.ipynb 확인
- [ ] 모든 셀이 순차적으로 실행 가능
- [ ] `!pip install` 명령어로 라이브러리 설치
- [ ] `./data/` 경로에서 데이터 로드
- [ ] `./submission.csv`로 결과 저장
- [ ] `aifactory.score.submit()` 함수 호출

### 3. API KEY 설정
- [ ] AI Factory 마이페이지에서 KEY 발급
- [ ] task.ipynb의 `MY_KEY` 변수에 입력

### 4. 불필요한 파일 제거
- [ ] `.git/` 폴더 제거
- [ ] `.venv/` 또는 가상환경 폴더 제거
- [ ] `__pycache__/` 폴더 제거
- [ ] 로컬 데이터 폴더 제거
- [ ] 개인정보 포함 파일 제거

---

## 🚀 제출 방법

### 방법 1: task.ipynb를 통한 자동 제출 (권장)

1. **API KEY 발급**
   ```
   AI Factory 마이페이지 → API KEY 발급
   ```

2. **task.ipynb 수정**
   ```python
   # 10. AI Factory 제출 셀에서
   MY_KEY = "발급받은-API-KEY"  # ← 여기 입력
   ```

3. **노트북 실행**
   - Jupyter Notebook 또는 VSCode에서 `task.ipynb` 열기
   - 모든 셀을 순차적으로 실행
   - 마지막 제출 셀까지 실행

4. **제출 확인**
   ```
   ✅ 제출 완료!
   ```
   메시지 확인

### 방법 2: 압축 파일 수동 제출

1. **프로젝트 압축**
   ```bash
   # 현재 폴더 및 하위 폴더를 압축
   zip -r submission.zip . \
     -x "*.git*" \
     -x "*__pycache__*" \
     -x "*.venv*" \
     -x "*data/*"
   ```

2. **AI Factory에 업로드**
   - 경진대회 페이지 → 제출 탭
   - 압축 파일 업로드

---

## ⚠️ 주의사항

### 1. 파일 크기
- 압축 파일이 너무 크면 업로드 시간이 오래 걸립니다
- 불필요한 파일을 제거하여 크기를 최소화하세요
- 가중치 파일만 포함하고, 학습 데이터는 제외하세요

### 2. 경로 사용
```python
# ✅ 좋은 예: 상대 경로
ckpt = './weights/MM-Det/current_model.pth'
data_dir = './data'

# ❌ 나쁜 예: 절대 경로
ckpt = '/home/user/workspace/weights/MM-Det/current_model.pth'
data_dir = '/workspace/data'
```

### 3. 추론 시간
- 최대 3시간 내에 추론 완료해야 함
- 배치 크기를 적절히 조정하여 시간 단축
- 불필요한 연산 최소화

### 4. 메모리 관리
```python
# GPU 메모리 절약 옵션
load_8bit = True   # 8bit 양자화
batch_size = 4     # 배치 크기 조정
```

### 5. submission.csv 형식
```python
# 올바른 형식
submission_df = pd.DataFrame({
    'filename': ['image1.jpg', 'video1.mp4'],  # 문자열
    'label': [0, 1]                             # 정수
})

# 데이터 타입 확인
submission_df['filename'] = submission_df['filename'].astype(str)
submission_df['label'] = submission_df['label'].astype(int)
```

---

## 🔧 문제 해결

### Q1. "모듈을 찾을 수 없습니다" 오류
```python
# task.ipynb의 첫 번째 셀에서 설치
!pip install -q [패키지명]
```

### Q2. "체크포인트를 찾을 수 없습니다" 오류
```bash
# 가중치 파일 경로 확인
ls -la weights/MM-Det/
ls -la weights/ViT/
```

### Q3. "API KEY가 유효하지 않습니다" 오류
- AI Factory 마이페이지에서 KEY 재발급
- KEY를 정확히 복사했는지 확인

### Q4. "추론 시간 초과" 오류
```python
# 배치 크기 줄이기
batch_size = 2  # 기본값: 4

# 양자화 사용
load_8bit = True
```

### Q5. submission.csv 형식 오류
```python
# 형식 확인
print(submission_df.head())
print(submission_df.dtypes)

# filename이 문자열, label이 정수인지 확인
```

---

## 📞 추가 지원

### 참고 문서
- [MM-Det README](./README.md)
- [AI Factory 경진대회 페이지](https://aifactory.space/)

### 문의
- 경진대회 Q&A 게시판 활용
- task.ipynb 내 주석 참고

---

## ✨ 제출 성공 확인

제출 후 다음을 확인하세요:
1. ✅ 제출 완료 메시지 수신
2. ✅ 경진대회 페이지에서 제출 이력 확인
3. ✅ 점수 산출 (Macro F1-score)

**행운을 빕니다! 🍀**
