#!/bin/bash

# AI Factory 제출 준비 스크립트
# 이 스크립트는 제출용 압축 파일을 생성합니다.

echo "======================================"
echo "AI Factory 제출 준비 스크립트"
echo "======================================"
echo ""

# 현재 디렉토리 확인
if [ ! -f "task.ipynb" ]; then
    echo "❌ 오류: task.ipynb 파일을 찾을 수 없습니다."
    echo "프로젝트 루트 디렉토리에서 실행하세요."
    exit 1
fi

echo "✅ task.ipynb 파일 확인 완료"

# 필수 파일 확인
echo ""
echo "필수 파일 확인 중..."

check_file() {
    if [ -e "$1" ]; then
        echo "  ✅ $1"
        return 0
    else
        echo "  ❌ $1 (없음)"
        return 1
    fi
}

all_files_exist=true

check_file "task.ipynb" || all_files_exist=false
check_file "weights/MM-Det/current_model.pth" || all_files_exist=false
check_file "weights/ViT/vit_base_r50_s16_224.orig_in21k" || all_files_exist=false
check_file "LLaVA/" || all_files_exist=false
check_file "models/" || all_files_exist=false
check_file "dataset/" || all_files_exist=false
check_file "utils/" || all_files_exist=false

if [ "$all_files_exist" = false ]; then
    echo ""
    echo "⚠️ 경고: 일부 필수 파일이 없습니다."
    echo "계속 진행하시겠습니까? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "취소되었습니다."
        exit 1
    fi
fi

# 불필요한 파일 확인
echo ""
echo "불필요한 파일 확인 중..."

cleanup_needed=false

if [ -d ".git" ]; then
    echo "  ⚠️ .git/ 폴더가 있습니다 (제외 권장)"
    cleanup_needed=true
fi

if [ -d ".venv" ]; then
    echo "  ⚠️ .venv/ 폴더가 있습니다 (제외 권장)"
    cleanup_needed=true
fi

if [ -d "__pycache__" ]; then
    echo "  ⚠️ __pycache__/ 폴더가 있습니다 (제외 권장)"
    cleanup_needed=true
fi

# 압축 파일명 설정
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="submission_${TIMESTAMP}.zip"

echo ""
echo "======================================"
echo "압축 파일 생성 중..."
echo "======================================"
echo "출력 파일: $OUTPUT_FILE"
echo ""

# 압축 생성
zip -r "$OUTPUT_FILE" . \
    -x "*.git/*" \
    -x "*__pycache__/*" \
    -x "*.venv/*" \
    -x "*venv/*" \
    -x "*.pyc" \
    -x "*data/*" \
    -x "*.DS_Store" \
    -x "*submission_*.zip" \
    -x "*.ipynb_checkpoints/*" \
    -q

if [ $? -eq 0 ]; then
    echo "✅ 압축 완료!"
    echo ""
    echo "======================================"
    echo "제출 파일 정보"
    echo "======================================"

    # 파일 크기 확인
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "파일명: $OUTPUT_FILE"
    echo "크기: $FILE_SIZE"

    # 파일 개수 확인
    FILE_COUNT=$(unzip -l "$OUTPUT_FILE" | tail -1 | awk '{print $2}')
    echo "포함된 파일 수: $FILE_COUNT"

    echo ""
    echo "======================================"
    echo "다음 단계"
    echo "======================================"
    echo "1. task.ipynb 열기"
    echo "2. API KEY 설정 (MY_KEY 변수)"
    echo "3. 모든 셀 실행"
    echo "4. AI Factory에 제출"
    echo ""
    echo "또는"
    echo ""
    echo "1. $OUTPUT_FILE 파일을"
    echo "2. AI Factory 경진대회 페이지에 업로드"
    echo ""
    echo "✅ 준비 완료!"
else
    echo "❌ 압축 실패!"
    exit 1
fi
