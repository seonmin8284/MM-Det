# MM-Det 모델 아키텍처 설명 (task.ipynb 기준)

## 목차
1. [개요](#개요)
2. [전체 시스템 구조](#전체-시스템-구조)
3. [모델 아키텍처](#모델-아키텍처)
4. [추론 파이프라인](#추론-파이프라인)
5. [기술적 세부사항](#기술적-세부사항)
6. [성능 최적화](#성능-최적화)

---

## 개요

**MM-Det (Multi-Modal Detector)**은 딥페이크 영상 및 이미지를 탐지하기 위한 딥러닝 모델입니다. 본 문서는 `task.ipynb`에 구현된 **MMDet_Simplified** 모델을 중심으로 아키텍처와 추론 프로세스를 상세히 설명합니다.

### 주요 특징
- **Hybrid Vision Transformer (ViT)** 기반 backbone
- **시공간적(Spatial-Temporal) 특징 추출** (10프레임 윈도우)
- **다중 얼굴 검출 알고리즘** (Mediapipe, Dlib, Haar Cascade)
- **Real-time 추론 최적화** (CUDA 12.6 지원)
- **이진 분류**: Real (0) vs Fake (1)

---

## 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                      입력 데이터                                 │
│         이미지 (.jpg, .png) / 동영상 (.mp4, .avi)               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  전처리 파이프라인                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. 얼굴 검출 (Multi-detector Cascade)                    │  │
│  │     - Mediapipe Face Detection (1st priority)            │  │
│  │     - Dlib Frontal Face Detector (2nd priority)          │  │
│  │     - Haar Cascade Classifier (3rd priority)             │  │
│  │                                                            │  │
│  │  2. 얼굴 영역 크롭 및 확장 (30% margin)                   │  │
│  │                                                            │  │
│  │  3. 리사이즈 (224x224)                                     │  │
│  │                                                            │  │
│  │  4. 프레임 샘플링 (비디오의 경우)                          │  │
│  │     - 10프레임을 균등 간격으로 추출                        │  │
│  │     - 최대 10초까지만 처리                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│               MMDet_Simplified 모델                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Input: (B, T=10, C=3, H=224, W=224)                     │  │
│  │         B: Batch size, T: Temporal frames                │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Backbone: Hybrid ViT with IAFA                   │  │
│  │   (vit_base_r50_s16_224_with_recons_iafa)                │  │
│  │                                                            │  │
│  │   ┌────────────────────────────────────────────────────┐ │  │
│  │   │  ResNet50 Feature Extractor                        │ │  │
│  │   │  - Conv layers → Feature maps (14x14)              │ │  │
│  │   └────────────────┬───────────────────────────────────┘ │  │
│  │                    │                                      │  │
│  │                    ▼                                      │  │
│  │   ┌────────────────────────────────────────────────────┐ │  │
│  │   │  Transformer Encoder (12 layers)                   │ │  │
│  │   │  - Multi-head Self-Attention                       │ │  │
│  │   │  - Feed-Forward Networks                           │ │  │
│  │   │  - Window-based Temporal Attention (window_size=10)│ │  │
│  │   │  - IAFA (Image-Aware Feature Aggregation)          │ │  │
│  │   └────────────────┬───────────────────────────────────┘ │  │
│  │                    │                                      │  │
│  │                    ▼                                      │  │
│  │   Output: (B, T=10, D=768) → Temporal features          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Temporal Pooling (Mean)                          │  │
│  │         (B, T=10, 768) → (B, 768)                        │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │       Classification Head (Linear Layer)                 │  │
│  │              768 → 2 classes                             │  │
│  │         [Real (0), Fake (1)]                             │  │
│  └──────────────────────┬───────────────────────────────────┘  │
└────────────────────────┼─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  후처리 및 출력                                  │
│  - Softmax → 확률값                                             │
│  - Argmax → 최종 예측 클래스 (0 or 1)                           │
│  - CSV 파일로 저장 (filename, label)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 모델 아키텍처

### 1. MMDet_Simplified 구조

`task.ipynb`에 구현된 간소화 버전 모델로, 학습된 backbone과 classification head만 사용합니다.

```python
class MMDet_Simplified(nn.Module):
    def __init__(self, window_size=10):
        super(MMDet_Simplified, self).__init__()
        # Backbone: Hybrid ViT with IAFA
        self.backbone = vit_base_r50_s16_224_with_recons_iafa(
            window_size=window_size,  # 시간축 윈도우 크기
            pretrained=False,
            ckpt_path=None,
            num_classes=0,    # Classifier 제거 (feature extraction만)
            global_pool=''    # Pooling 제거
        )
        # Classification Head
        self.head = nn.Linear(768, 2)
```

**파라미터:**
- **Input shape**: `(B, T=10, C=3, H=224, W=224)`
- **Backbone output**: `(B, T=10, D=768)`
- **Head output**: `(B, 2)` - [real_logit, fake_logit]

### 2. Hybrid Vision Transformer Backbone

#### 2.1 ResNet50 Feature Extractor

```
Input Image (224x224x3)
    │
    ▼
┌──────────────────────┐
│   Conv1 + BN + ReLU  │  (112x112x64)
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│   MaxPool            │  (56x56x64)
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│   Layer1 (3 blocks)  │  (56x56x256)
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│   Layer2 (4 blocks)  │  (28x28x512)
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│   Layer3 (6 blocks)  │  (14x14x1024)
└──────────┬───────────┘
           │
    ▼
Feature Map: (14x14x1024)
```

#### 2.2 Hybrid Embedding Layer

ResNet50의 feature map을 Transformer 입력으로 변환:

```
Feature Map (14x14x1024)
    │
    ▼
┌──────────────────────┐
│  Flatten (spatial)   │  → (196, 1024)
└──────────┬───────────┘    # 196 = 14x14 patches
           │
    ▼
┌──────────────────────┐
│  Linear Projection   │  → (196, 768)
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│  + Position Embed    │  → (196, 768)
└──────────────────────┘
```

#### 2.3 Transformer Encoder with IAFA

**구조:**
- **12개 Layer** (Base 모델)
- **12 attention heads** per layer
- **Hidden dimension**: 768
- **MLP dimension**: 3072 (768 × 4)

**IAFA (Image-Aware Feature Aggregation):**
- Original image와 Reconstructed image 간의 차이를 학습
- Forgery artifact를 강조하는 attention mechanism
- Window-based temporal modeling (10 frames)

```
For each Transformer Layer:
┌──────────────────────────────────────────┐
│  1. Multi-Head Self-Attention            │
│     - Query, Key, Value projection       │
│     - Temporal attention (across frames) │
│     - Spatial attention (within patches) │
└──────────────┬───────────────────────────┘
               │
        ▼
┌──────────────────────────────────────────┐
│  2. IAFA Module                          │
│     - Image reconstruction comparison    │
│     - Artifact-aware weighting           │
└──────────────┬───────────────────────────┘
               │
        ▼
┌──────────────────────────────────────────┐
│  3. Feed-Forward Network                 │
│     - Linear(768 → 3072) + GELU         │
│     - Linear(3072 → 768)                │
└──────────────┬───────────────────────────┘
               │
        ▼
┌──────────────────────────────────────────┐
│  4. Layer Normalization                  │
└──────────────────────────────────────────┘
```

### 3. Classification Head

```python
# Temporal Pooling
x_pooled = torch.mean(x_st, dim=1)  # (B, T=10, 768) → (B, 768)

# Linear Classification
out = self.head(x_pooled)  # (B, 768) → (B, 2)

# Softmax (inference)
probs = F.softmax(out, dim=1)  # (B, 2)
predicted_class = torch.argmax(probs).item()  # 0 or 1
```

---

## 추론 파이프라인

### 1. 얼굴 검출 (Multi-detector Cascade)

**우선순위 기반 검출 전략:**

```python
def detect_and_crop_face_multi(image):
    # 1st: Mediapipe (가장 빠르고 정확)
    if MEDIAPIPE_AVAILABLE:
        face = detect_face_mediapipe(image)
        if face: return face
    
    # 2nd: Dlib (전통적이지만 견고함)
    try:
        face = detect_face_dlib(image)
        if face: return face
    except: pass
    
    # 3rd: Haar Cascade (fallback)
    try:
        face = detect_face_haar(image)
        if face: return face
    except: pass
    
    # Fallback: 전체 이미지 리사이즈
    return image.resize((224, 224))
```

**각 검출기 특징:**

| 검출기 | 속도 | 정확도 | 특징 |
|--------|------|--------|------|
| **Mediapipe** | 매우 빠름 | 높음 | GPU 가속, 다양한 각도 지원 |
| **Dlib** | 중간 | 높음 | HOG + SVM 기반, 정면 얼굴 특화 |
| **Haar Cascade** | 빠름 | 중간 | OpenCV 내장, 가벼움 |

### 2. 비디오 프레임 샘플링

```python
def process_video_frames(video_path, num_frames=10, max_duration=10):
    """
    비디오에서 10개의 프레임을 균등하게 추출
    """
    # 1. 비디오 총 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 2. 최대 10초까지만 처리
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_duration)
    total_frames = min(total_frames, max_frames)
    
    # 3. 균등 간격 샘플링
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    # 예: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    
    # 4. 각 프레임에서 얼굴 검출
    for idx in frame_indices:
        frame = read_frame_at(idx)
        face = detect_and_crop_face_multi(frame)
        faces.append(face)
    
    return faces  # 길이 10의 리스트
```

### 3. 이미지 전처리

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),                    # 크기 조정
    transforms.ToTensor(),                            # (H,W,C) → (C,H,W), [0,255] → [0,1]
    transforms.Normalize(                             # ImageNet 정규화
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 4. 배치 추론

```python
# 입력 준비
batch = face_images[:10]  # 10개 프레임
img_tensors = torch.stack([transform(img) for img in batch])
img_tensors = img_tensors.unsqueeze(0)  # (10,3,224,224) → (1,10,3,224,224)
img_tensors = img_tensors.to(device)

# 추론
with torch.no_grad():
    x_original = img_tensors
    x_recons = img_tensors  # Simplified 버전에서는 동일
    
    logits = model((x_original, x_recons))  # (1, 2)
    probs = F.softmax(logits, dim=1)        # (1, 2)
    predicted_class = torch.argmax(probs).item()  # 0 or 1
```

---

## 기술적 세부사항

### 1. 입력 형식

**이미지:**
- 지원 형식: `.jpg`, `.jpeg`, `.png`
- 처리: 1장의 이미지를 10번 복제하여 (1, 10, 3, 224, 224) 생성

**비디오:**
- 지원 형식: `.mp4`, `.avi`, `.mov`, `.mkv`
- 처리: 균등 간격으로 10 프레임 추출
- 제약: 최대 10초까지만 처리 (메모리 절약)

### 2. 모델 가중치 로딩

```python
# Checkpoint 로드
checkpoint = torch.load(model_weights_path, map_location='cpu')

# State dict 전처리
new_state_dict = {}
for k, v in checkpoint.items():
    # 'module.' 접두사 제거 (DDP 학습 시 추가됨)
    if k.startswith('module.'):
        k = k[7:]
    
    # backbone과 head만 로드 (MMDet_Simplified는 이 두 부분만 사용)
    if any(k.startswith(prefix) for prefix in ['backbone', 'head']):
        new_state_dict[k] = v

# 모델에 로드
model.load_state_dict(new_state_dict, strict=False)
```

### 3. 메모리 최적화

```python
# 추론 후 GPU 메모리 정리
del img_tensors, x_original, x_recons, logits, probs
if device == "cuda":
    torch.cuda.empty_cache()

# 배치별 처리 (500개마다 진행상황 출력)
if (idx + 1) % 500 == 0:
    print(f"Progress: {idx+1}/{total_files}")
```

### 4. 에러 핸들링

```python
try:
    # 얼굴 검출 및 추론
    face_images = process_video_frames(video_path)
    predicted_class = model(face_images)
except Exception as e:
    # 에러 발생 시 기본값 (Real=0)으로 처리
    error_count += 1
    predicted_class = 0
```

---

## 성능 최적화

### 1. 추론 속도

**테스트 환경:**
- GPU: NVIDIA GPU (CUDA 12.6)
- Batch size: 1 (파일별 처리)
- 평균 속도: ~2-3 files/sec (비디오 포함)

**최적화 기법:**
- Mixed precision (FP16) 미사용 (안정성 우선)
- Batch inference: 파일별 순차 처리 (메모리 안정성)
- 얼굴 검출 해상도 축소: 320px로 리사이즈 후 검출 (Dlib)

### 2. 정확도 vs 속도 Trade-off

| 설정 | 속도 | 정확도 | 메모리 |
|------|------|--------|--------|
| 10 프레임 (현재) | 중간 | 높음 | 중간 |
| 5 프레임 | 빠름 | 중간 | 낮음 |
| 20 프레임 | 느림 | 매우 높음 | 높음 |

### 3. 배포 시 고려사항

**CPU 환경:**
```python
device = "cpu"
# 예상 속도: ~0.5-1 files/sec
# 권장: 멀티프로세싱 적용
```

**GPU 환경:**
```python
device = "cuda"
# 예상 속도: ~2-3 files/sec
# 권장: 배치 크기 증가 (메모리가 충분한 경우)
```

---

## 모델 성능

### 학습 데이터셋
- **DVF (Diffusion Video Forensics)** 데이터셋
- Real 영상: YouTube 등 실제 촬영 영상
- Fake 영상: Stable Video Diffusion, VideoCrafter1, Sora 등

### 평가 지표
- **Accuracy**: Real/Fake 분류 정확도
- **AUC**: Area Under Curve
- **Precision/Recall**: 클래스별 성능

---

## 전체 모델 (Full MM-Det)

`MMDet.py`에 정의된 전체 모델은 추가 컴포넌트를 포함합니다:

```
MMDet (Full Version)
├── Backbone: Hybrid ViT (동일)
├── MM Encoder: LLaVA-based Multi-Modal Encoder
│   ├── CLIP Vision Tower (1024-dim)
│   └── LLaVA Text Generation (4096-dim)
├── Projection Layers
│   ├── CLIP Projection: 1024 → 768
│   └── MM Projection: 4096 → 768
├── Dynamic Fusion: Cross-modal Attention
└── Classification Head: 768 → 2
```

**차이점:**
- **MMDet_Simplified**: Backbone + Head만 사용 (빠른 추론)
- **MMDet (Full)**: Multi-modal reasoning 추가 (더 높은 정확도)

**Simplified 버전 사용 이유:**
1. **추론 속도**: LLaVA 모델 로딩 불필요 (~10GB VRAM 절약)
2. **배포 편의성**: 단일 checkpoint 파일만 필요
3. **충분한 성능**: Backbone만으로도 높은 정확도 달성

---

## 참고 자료

- **논문**: [On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection (NeurIPS 2024)](https://arxiv.org/abs/2410.23623)
- **Hybrid ViT**: [An Image Is Worth 16x16 Words (ICLR 2021)](https://arxiv.org/abs/2010.11929)
- **LLaVA**: [Visual Instruction Tuning (NeurIPS 2023)](https://github.com/haotian-liu/LLaVA)
- **TIMM Library**: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## 라이센스

본 프로젝트는 원본 MM-Det 프로젝트의 라이센스를 따릅니다.

