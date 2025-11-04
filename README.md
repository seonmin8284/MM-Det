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

## 성능 향상 전략

현재 MMDet_Simplified 모델의 성능을 더욱 개선하기 위한 체계적인 방법들을 소개합니다.

### 1. 모델 아키텍처 개선

#### 1.1 Full MM-Det 활용 (권장 ⭐)

**현재 사용:** MMDet_Simplified (Backbone + Head만)  
**업그레이드:** Full MM-Det (Multi-Modal Encoder 추가)

```python
# Full MM-Det으로 전환
model = MMDet(config)  # MMDet_Simplified 대신

# 장점:
# - LLaVA 기반 시각-언어 추론 활용
# - CLIP 비전 특징 (1024-dim) 추가
# - 텍스트 기반 forgery 설명 생성 (4096-dim)
# - Dynamic Fusion으로 cross-modal attention
```

**예상 성능 향상:**
- Accuracy: +3~5%
- 특히 복잡한 diffusion artifact 탐지에서 효과적
- 단점: 추론 시간 증가 (~5-10배), VRAM 10GB+ 필요

**구현 방법:**
```python
config = {
    'lmm_ckpt': './weights/llava-7b-1.5-rfrd',
    'lmm_base': None,
    'load_8bit': False,
    'load_4bit': False,
    'conv_mode': 'llava_v1',
    'new_tokens': 256,
    'selected_layers': [-1],
    'interval': 10,
    'cache_mm': False,  # MM Encoder 활성화
    # ... 기타 설정
}
model = MMDet(config)
```

#### 1.2 더 큰 Backbone 사용

**현재:** `vit_base_r50_s16_224` (768-dim)  
**업그레이드 옵션:**

| 모델 | Params | Hidden Dim | 예상 성능 | VRAM |
|------|--------|------------|-----------|------|
| ViT-Base (현재) | 90M | 768 | Baseline | 4GB |
| **ViT-Large** | 304M | 1024 | +2~4% | 8GB |
| ViT-Huge | 632M | 1280 | +3~6% | 16GB+ |

```python
# ViT-Large로 변경
from .vit.stv_transformer_hybrid import vit_large_r50_s16_224_with_recons_iafa

self.backbone = vit_large_r50_s16_224_with_recons_iafa(
    window_size=window_size,
    pretrained=True
)
self.head = nn.Linear(1024, 2)  # 768 → 1024
```

#### 1.3 Ensemble 모델

**여러 모델의 예측을 결합하여 성능 향상:**

```python
class EnsembleMMDet(nn.Module):
    def __init__(self):
        super().__init__()
        # 다양한 설정의 모델들
        self.model1 = MMDet_Simplified(window_size=10)
        self.model2 = MMDet_Simplified(window_size=15)  # 더 긴 temporal window
        self.model3 = MMDet(config)  # Full 버전
        
    def forward(self, x):
        # Soft voting
        out1 = F.softmax(self.model1(x), dim=1)
        out2 = F.softmax(self.model2(x), dim=1)
        out3 = F.softmax(self.model3(x), dim=1)
        
        # 가중 평균 (성능 기반 가중치)
        ensemble_out = 0.3 * out1 + 0.3 * out2 + 0.4 * out3
        return ensemble_out
```

**예상 성능 향상:** +2~3% (단, 추론 시간 3배 증가)

---

### 2. 데이터 전처리 개선

#### 2.1 더 많은 프레임 사용

**현재:** 10 프레임  
**개선안:** 15~20 프레임 (더 풍부한 temporal 정보)

```python
# process_video_frames() 수정
num_frames_to_extract = 20  # 10 → 20

# 모델도 수정 필요
model = MMDet_Simplified(window_size=20)
```

**Trade-off:**
- 장점: 시간적 일관성 검증 강화 (+1~2% accuracy)
- 단점: 추론 시간 2배, 메모리 2배

#### 2.2 Multi-Scale 추론

**다양한 해상도에서 추론 후 결합:**

```python
def multi_scale_inference(image, model):
    scales = [224, 256, 288]
    predictions = []
    
    for scale in scales:
        # 각 스케일로 리사이즈
        resized = transforms.Resize((scale, scale))(image)
        # Center crop to 224x224
        cropped = transforms.CenterCrop(224)(resized)
        
        # 추론
        pred = model(cropped)
        predictions.append(pred)
    
    # 평균
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred
```

**예상 성능 향상:** +1~2%

#### 2.3 얼굴 검출 개선

**현재 방식의 문제점:**
- 얼굴이 검출되지 않으면 전체 이미지 사용
- 측면/기울어진 얼굴 놓침

**개선 방안:**

```python
# 1. 더 강력한 얼굴 검출 모델 사용
# MTCNN, RetinaFace, YOLO-Face 등

# 2. Multi-face 처리
def detect_all_faces(image):
    """모든 얼굴 검출 → 가장 큰 얼굴 선택"""
    faces = detector.detect_multi(image)
    if len(faces) == 0:
        return None
    # 가장 큰 얼굴 반환
    largest_face = max(faces, key=lambda f: f.area)
    return crop_face(image, largest_face)

# 3. Face alignment 추가
def align_face(face_image):
    """얼굴 랜드마크 기반 정렬"""
    landmarks = get_landmarks(face_image)
    aligned = affine_transform(face_image, landmarks)
    return aligned
```

**예상 성능 향상:** +2~3% (특히 다양한 각도의 얼굴에서)

#### 2.4 데이터 증강 (학습 시)

**학습 시 적용할 augmentation:**

```python
train_transform = transforms.Compose([
    # 기하학적 변환
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    
    # 색상 변환
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    
    # Diffusion artifact 강조
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    
    # 정규화
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    
    # CutOut / Random Erasing
    transforms.RandomErasing(p=0.2),
])
```

**예상 성능 향상:** +3~5% (학습 데이터가 제한적일 때 효과적)

---

### 3. 학습 전략 최적화

#### 3.1 학습률 스케줄링

**현재 문제:** 고정 학습률 사용 시 최적점 근처에서 진동

**개선 방안:**

```python
# Cosine Annealing with Warm Restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # 첫 restart까지 epoch 수
    T_mult=2,    # restart 주기 증가 배수
    eta_min=1e-6 # 최소 학습률
)

# 또는 ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True
)
```

#### 3.2 Loss Function 개선

**현재:** Cross Entropy Loss (단순 분류)

**개선 옵션:**

```python
# 1. Focal Loss (hard example에 집중)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# 2. Label Smoothing (과적합 방지)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 3. ArcFace Loss (feature discrimination 강화)
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        self.m = m
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        # Cosine similarity
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        # Add margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        # Apply
        one_hot = F.one_hot(labels, num_classes=2)
        output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)
```

**예상 성능 향상:**
- Focal Loss: +1~2% (클래스 불균형이 있을 때)
- Label Smoothing: 과적합 감소, 일반화 성능 향상
- ArcFace: +2~3% (feature space에서 더 명확한 분리)

#### 3.3 Hard Negative Mining

**어려운 샘플에 집중하여 학습:**

```python
def hard_negative_mining(model, dataloader, ratio=0.3):
    """
    가장 분류하기 어려운 샘플들을 선별
    """
    model.eval()
    samples_with_loss = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            losses = F.cross_entropy(outputs, labels, reduction='none')
            
            for i, loss in enumerate(losses):
                samples_with_loss.append((inputs[i], labels[i], loss.item()))
    
    # Loss 기준 내림차순 정렬
    samples_with_loss.sort(key=lambda x: x[2], reverse=True)
    
    # 상위 ratio% 샘플만 선택
    num_hard = int(len(samples_with_loss) * ratio)
    hard_samples = samples_with_loss[:num_hard]
    
    return hard_samples

# 학습 루프에서
for epoch in range(num_epochs):
    # 일반 학습
    train_epoch(model, train_loader)
    
    # Hard negative mining (3 epoch마다)
    if epoch % 3 == 0:
        hard_samples = hard_negative_mining(model, train_loader)
        train_on_hard_samples(model, hard_samples)
```

**예상 성능 향상:** +2~3%

#### 3.4 Knowledge Distillation

**큰 모델의 지식을 작은 모델로 전달:**

```python
# Teacher: Full MM-Det (큰 모델)
teacher = MMDet(config)
teacher.load_state_dict(torch.load('teacher_weights.pth'))
teacher.eval()

# Student: MMDet_Simplified (작은 모델)
student = MMDet_Simplified()

# Distillation Loss
def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    # Soft target loss
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard target loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * soft_loss + (1 - alpha) * hard_loss

# 학습
for inputs, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(inputs)
    
    student_logits = student(inputs)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

**장점:**
- Simplified 모델의 성능을 Full 모델 수준으로 근접
- 추론 시에는 빠른 Student 모델만 사용
- 예상 성능 향상: +2~4%

---

### 4. Test-Time 최적화

#### 4.1 Test-Time Augmentation (TTA)

**추론 시 데이터 증강을 적용하여 예측 안정성 향상:**

```python
def test_time_augmentation(model, image, num_augments=5):
    """
    여러 augmented 버전에 대해 추론 후 평균
    """
    predictions = []
    
    # Original
    pred = model(image)
    predictions.append(pred)
    
    # Horizontal flip
    pred_flip = model(transforms.functional.hflip(image))
    predictions.append(pred_flip)
    
    # Multi-scale
    for scale in [0.9, 1.0, 1.1]:
        h, w = int(224 * scale), int(224 * scale)
        resized = transforms.functional.resize(image, (h, w))
        cropped = transforms.functional.center_crop(resized, 224)
        pred_scale = model(cropped)
        predictions.append(pred_scale)
    
    # 평균 (Soft voting)
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred
```

**예상 성능 향상:** +1~2% (약간의 추론 시간 증가)

#### 4.2 Temporal Consistency Check (비디오)

**연속된 프레임 간의 일관성 검증:**

```python
def temporal_consistency_check(model, video_frames, threshold=0.3):
    """
    프레임별 예측의 분산이 크면 재검토
    """
    frame_predictions = []
    
    # 각 프레임 개별 예측
    for frame in video_frames:
        pred = model(frame.unsqueeze(0))
        prob_fake = F.softmax(pred, dim=1)[0, 1].item()
        frame_predictions.append(prob_fake)
    
    # 분산 계산
    variance = np.var(frame_predictions)
    
    if variance > threshold:
        # 분산이 크면 → 더 많은 프레임 샘플링하여 재추론
        more_frames = sample_more_frames(video, num_frames=20)
        return model(more_frames)
    else:
        # 일관성 있으면 → 평균 사용
        return np.mean(frame_predictions)
```

**예상 성능 향상:** +1~2% (비디오 데이터에서)

#### 4.3 Confidence-based Filtering

**낮은 confidence 샘플에 대해 재처리:**

```python
def confidence_based_inference(model, inputs, confidence_threshold=0.7):
    """
    Confidence가 낮으면 추가 처리
    """
    # 1차 추론
    logits = model(inputs)
    probs = F.softmax(logits, dim=1)
    confidence = torch.max(probs, dim=1)[0].item()
    prediction = torch.argmax(probs).item()
    
    if confidence < confidence_threshold:
        # Low confidence → TTA 적용
        tta_pred = test_time_augmentation(model, inputs)
        tta_probs = F.softmax(tta_pred, dim=1)
        prediction = torch.argmax(tta_probs).item()
    
    return prediction
```

---

### 5. 시스템 레벨 최적화

#### 5.1 배치 처리

**현재:** 파일별 순차 처리 (batch_size=1)  
**개선:** 배치 단위 처리

```python
def batch_inference(model, file_list, batch_size=8):
    """
    여러 파일을 배치로 묶어서 처리
    """
    results = []
    dataloader = DataLoader(
        dataset=DeepfakeDataset(file_list),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch_inputs, filenames in dataloader:
            batch_inputs = batch_inputs.to(device)
            outputs = model(batch_inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            for filename, pred in zip(filenames, predictions):
                results.append((filename, pred.item()))
    
    return results
```

**성능 향상:**
- 추론 속도: 2~4배 증가 (GPU 활용도 향상)
- 정확도: 동일

#### 5.2 Mixed Precision Inference

**FP16 사용으로 메모리 및 속도 개선:**

```python
from torch.cuda.amp import autocast

# 추론 시
with torch.no_grad():
    with autocast():  # FP16 사용
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
```

**성능 향상:**
- 추론 속도: 1.5~2배 증가
- 메모리: 50% 감소
- 정확도: -0.1~0% (거의 영향 없음)

#### 5.3 모델 경량화

**ONNX 또는 TensorRT로 변환:**

```python
# PyTorch → ONNX
dummy_input = torch.randn(1, 10, 3, 224, 224).to(device)
torch.onnx.export(
    model,
    (dummy_input, dummy_input),
    "mmdet_simplified.onnx",
    input_names=['original', 'recons'],
    output_names=['output'],
    dynamic_axes={
        'original': {0: 'batch_size'},
        'recons': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# ONNX Runtime으로 추론
import onnxruntime as ort
session = ort.InferenceSession("mmdet_simplified.onnx")
outputs = session.run(None, {
    'original': inputs_numpy,
    'recons': inputs_numpy
})
```

**성능 향상:**
- 추론 속도: 2~3배 증가
- 정확도: 동일

---

### 6. 하이브리드 접근법 (권장 ⭐⭐⭐)

**여러 전략을 조합한 최적 구성:**

```python
class OptimizedMMDet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 더 큰 backbone (ViT-Large)
        self.backbone = vit_large_r50_s16_224_with_recons_iafa(
            window_size=15,  # 10 → 15 프레임
            pretrained=True
        )
        self.head = nn.Linear(1024, 2)
    
    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        features = torch.mean(features, dim=1)
        
        # Classification
        logits = self.head(features)
        return logits

# 학습 설정
model = OptimizedMMDet()

# Focal Loss + Label Smoothing
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Cosine scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# 학습 시 데이터 증강
train_loader = DataLoader(
    dataset=DeepfakeDataset(
        transform=get_strong_augmentation()
    ),
    batch_size=16,  # 배치 크기 증가
    num_workers=8,
    pin_memory=True
)

# 추론 시
def optimized_inference(model, inputs):
    # TTA 적용
    preds = test_time_augmentation(model, inputs, num_augments=5)
    
    # Confidence check
    confidence = torch.max(F.softmax(preds, dim=1)).item()
    if confidence < 0.7:
        # Low confidence → 더 많은 프레임 사용
        inputs_extended = extend_frames(inputs, num_frames=20)
        preds = model(inputs_extended)
    
    return torch.argmax(preds).item()
```

**예상 전체 성능 향상:** +5~10%

**구현 우선순위:**
1. **Full MM-Det 활용** (즉시 +3~5%)
2. **ViT-Large backbone** (비교적 쉬움, +2~4%)
3. **TTA + Ensemble** (추론 시 적용, +2~3%)
4. **Better Loss Function** (학습 시, +1~2%)
5. **데이터 증강** (학습 시, +3~5%)

---

### 7. 성능 벤치마크 비교

| 방법 | 예상 Accuracy 향상 | 추론 시간 | 메모리 | 구현 난이도 |
|------|-------------------|-----------|--------|------------|
| **Baseline (현재)** | - | 1x | 4GB | - |
| Full MM-Det | +3~5% | 8x | 14GB | 쉬움 |
| ViT-Large | +2~4% | 2x | 8GB | 쉬움 |
| Ensemble (3 models) | +2~3% | 3x | 12GB | 중간 |
| TTA (5 augments) | +1~2% | 1.5x | 4GB | 쉬움 |
| 더 많은 프레임 (20) | +1~2% | 2x | 8GB | 쉬움 |
| Focal Loss | +1~2% | 1x | 4GB | 쉬움 |
| Knowledge Distillation | +2~4% | 1x | 4GB | 중간 |
| **Hybrid (추천)** | **+7~12%** | 3x | 10GB | 중간 |

**최적 ROI (Return on Investment) 조합:**
```
Full MM-Det + TTA + Better Loss
→ 예상 향상: +5~8%
→ 추론 시간: ~10x
→ 메모리: ~14GB
→ 구현: 쉬움~중간
```

**빠른 배포용 조합 (속도 우선):**
```
ViT-Large + Focal Loss + Multi-scale input
→ 예상 향상: +4~6%
→ 추론 시간: ~2x
→ 메모리: ~8GB
→ 구현: 쉬움
```

---

## 참고 자료

- **논문**: [On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection (NeurIPS 2024)](https://arxiv.org/abs/2410.23623)
- **Hybrid ViT**: [An Image Is Worth 16x16 Words (ICLR 2021)](https://arxiv.org/abs/2010.11929)
- **LLaVA**: [Visual Instruction Tuning (NeurIPS 2023)](https://github.com/haotian-liu/LLaVA)
- **TIMM Library**: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## 라이센스

본 프로젝트는 원본 MM-Det 프로젝트의 라이센스를 따릅니다.

