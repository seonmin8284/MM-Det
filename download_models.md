# 모델 다운로드 가이드

현재 프로젝트에 모델 가중치 파일이 없습니다. 다음 단계에 따라 다운로드하세요.

## 필요한 모델 파일

### 1. MM-Det 메인 모델
- **다운로드 링크**: https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing
- **파일**: `MM-Det/current_model.pth`
- **저장 위치**: `./weights/MM-Det/current_model.pth`

### 2. ViT Backbone 사전학습 모델
- **다운로드 링크**: https://drive.google.com/drive/folders/1RRNS8F7ETZWrcBu8fvB3pM9qHbmSEEzy?usp=sharing
- **파일**: `ViT/vit_base_r50_s16_224.orig_in21k`
- **저장 위치**: `./weights/ViT/vit_base_r50_s16_224.orig_in21k`

### 3. LLaVA 모델 (HuggingFace에서 자동 다운로드)
- **모델**: sparklexfantasy/llava-7b-1.5-rfrd
- **자동 다운로드**: 첫 실행시 자동으로 다운로드됨

## 다운로드 후 폴더 구조

```
weights/
├── MM-Det/
│   └── current_model.pth          # 다운로드 필요
├── ViT/
│   └── vit_base_r50_s16_224.orig_in21k  # 다운로드 필요
├── config.json
├── tokenizer.model
└── (기타 tokenizer 파일들)
```

## 빠른 다운로드 (wget 사용)

Google Drive 직접 다운로드가 어려운 경우, 파일 ID를 사용하여 다운로드할 수 있습니다:

```bash
# MM-Det 폴더 생성
mkdir -p weights/MM-Det
mkdir -p weights/ViT

# Google Drive에서 다운로드 (파일 ID 필요)
# 아래는 예시입니다. 실제 파일 ID로 교체하세요.
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILE_ID" -O weights/MM-Det/current_model.pth && rm -rf /tmp/cookies.txt
```

## 다운로드 확인

```bash
# 파일 존재 확인
ls -lh weights/MM-Det/current_model.pth
ls -lh weights/ViT/vit_base_r50_s16_224.orig_in21k

# 파일 크기 확인 (예상)
# current_model.pth: ~수백 MB
# vit_base_r50_s16_224.orig_in21k: ~수백 MB
```

## 대안: 더미 모델로 구조만 테스트

실제 모델 없이 제출 구조만 테스트하려면:

```bash
# 더미 파일 생성 (테스트용)
mkdir -p weights/MM-Det
mkdir -p weights/ViT
touch weights/MM-Det/current_model.pth
touch weights/ViT/vit_base_r50_s16_224.orig_in21k
```

⚠️ **주의**: 더미 파일로는 실제 추론이 불가능합니다!
