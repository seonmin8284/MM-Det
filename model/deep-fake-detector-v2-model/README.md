# Deep Fake Detector V2 Model

이 폴더는 MM-Det 딥페이크 감지 모델의 추론 결과를 저장하는 폴더입니다.

## 폴더 구조

```
deep-fake-detector-v2-model/
├── README.md                 # 이 파일
├── detection_results.csv    # 추론 결과 (task.ipynb 실행 후 생성됨)
├── model_info.json          # 모델 정보 (task.ipynb 실행 후 생성됨)
└── submission.csv           # 제출용 파일 (task.ipynb 실행 후 생성됨)
```

## 결과 파일 설명

### detection_results.csv
각 데이터셋별 성능 지표가 포함되어 있습니다:
- dataset: 데이터셋 이름
- accuracy: 정확도
- precision: 정밀도
- recall: 재현율
- f1_score: F1 점수
- auc: AUC 점수

### model_info.json
실험에 사용된 모델 정보:
- model_name: 모델 이름
- checkpoint: 사용된 체크포인트 경로
- lmm_model: Large Multi-Modal 모델
- pytorch_version: PyTorch 버전
- cuda_version: CUDA 버전
- gpu: 사용된 GPU
- test_datasets: 테스트한 데이터셋 목록

## 사용 방법

1. 상위 폴더의 `task.ipynb` 노트북을 실행하세요
2. 모든 셀을 순차적으로 실행하면 이 폴더에 결과가 저장됩니다
3. 생성된 결과 파일을 확인하세요

## 주의사항

- 이 폴더는 추론 결과 저장용이므로 직접 수정하지 마세요
- 모델 가중치는 `../../weights/` 폴더에 있습니다
- 데이터는 `../../data/` 폴더에 있습니다
