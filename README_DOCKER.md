# MM-Det Docker Setup

이 프로젝트를 Docker 컨테이너로 실행하기 위한 가이드입니다.

## 필수 조건

- Docker 설치 (https://www.docker.com/get-started)
- Docker Compose 설치 (Docker Desktop에 포함됨)

## 빌드 방법

### 1. Docker 이미지 빌드

```bash
docker build -t mm-det:latest .
```

또는 Docker Compose 사용:

```bash
docker-compose build
```

## 실행 방법

### 1. Docker Compose로 실행 (권장)

```bash
docker-compose up -d
```

### 2. 컨테이너 접속

```bash
docker-compose exec mm-det bash
```

또는:

```bash
docker exec -it mm-det-container bash
```

### 3. 직접 Docker 명령어로 실행

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/expts:/app/expts \
  -v $(pwd)/outputs:/app/outputs \
  mm-det:latest
```

## 컨테이너 내에서 실행

컨테이너에 접속한 후:

### 1. MM Representation 생성

```bash
python prepare_mm_cpu.py \
  --cached-data-root data/my_test_recons \
  --output-dir data/my_test_mm_representations \
  --lmm-ckpt liuhaotian/llava-v1.5-7b
```

### 2. 추론 실행

```bash
python test.py \
  --classes test \
  --ckpt ./weights/current_model.pth \
  --data-root ./data/my_test_recons \
  --cache-mm \
  --mm-root ./data/my_test_mm_representations \
  --sample-size -1
```

### 3. 비디오 전처리 (VQVAE reconstruction)

```bash
python prepare_reconstructed_dataset.py \
  -d data/my_test_video \
  -o data/my_test_recons \
  --device cpu
```

## 볼륨 마운트

Docker Compose는 다음 디렉토리들을 자동으로 마운트합니다:

- `./data` → `/app/data` - 데이터셋
- `./weights` → `/app/weights` - 모델 체크포인트
- `./expts` → `/app/expts` - 실험 결과
- `./outputs` → `/app/outputs` - 출력 파일

## 주의사항

1. **CPU 전용**: 이 컨테이너는 CPU 버전 PyTorch를 사용합니다 (RTX 5070 Ti 호환성 문제로 인해)

2. **LLaVA 모델**: 첫 실행 시 HuggingFace에서 자동으로 다운로드됩니다
   - `liuhaotian/llava-v1.5-7b` (~13GB)
   - 캐시 위치: `~/.cache/huggingface/`

3. **CLIP 모델**: `weights/clip-vit-large-patch14-336` 디렉토리에 있어야 합니다

4. **메모리 요구사항**: 최소 16GB RAM 권장

## 컨테이너 관리

### 컨테이너 중지

```bash
docker-compose down
```

### 컨테이너 재시작

```bash
docker-compose restart
```

### 로그 확인

```bash
docker-compose logs -f mm-det
```

### 컨테이너 삭제

```bash
docker-compose down -v
```

### 이미지 삭제

```bash
docker rmi mm-det:latest
```

## 문제 해결

### HuggingFace 다운로드 느릴 때

컨테이너 내에서:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 메모리 부족

Docker Desktop 설정에서 메모리 할당량을 늘리세요 (최소 8GB, 권장 16GB)

### 권한 문제 (Linux/Mac)

```bash
sudo chown -R $USER:$USER data weights expts outputs
```

## 추가 정보

- 모든 Python 패키지는 `requirements.txt`에 정의되어 있습니다
- 컨테이너는 `/app` 디렉토리에서 작동합니다
- CUDA는 비활성화되어 있습니다 (`CUDA_VISIBLE_DEVICES=""`)
