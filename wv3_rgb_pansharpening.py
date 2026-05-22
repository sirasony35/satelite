import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
import glob


def wv3_masked_pansharpen(ms_file, pan_file, output_file, filename_context):
    """
    외곽 배경 노이즈(블루/회색)를 검은색(0)으로 강제 마스킹하고
    동적 밴드 정렬 및 Cubic 융합을 처리하는 정밀 전처리 자동화 함수
    """
    # 1. 팬크로매틱 고해상도 메타데이터 및 밝기값 로드
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        pan_data = pan_src.read(1).astype('float32')
        out_shape = (pan_src.height, pan_src.width)

    # 2. 파일명 컨텍스트를 분석하여 SM_07 파일의 밴드 뒤틀림 예외 처리 설정
    if "SM_07" in filename_context.upper():
        red_band_idx = 5
        green_band_idx = 2
        blue_band_idx = 3
        print(f"-> [배경정제 및 밴드교정] {filename_context}: R=5, G=2, B=3 적용")
    else:
        red_band_idx = 5
        green_band_idx = 3
        blue_band_idx = 2

    # 3. 정해진 인덱스에 따라 3차 회선(Cubic) 리샘플링 데이터 로드
    with rasterio.open(ms_file) as ms_src:
        red = ms_src.read(red_band_idx, out_shape=out_shape, resampling=Resampling.cubic).astype('float32')
        green = ms_src.read(green_band_idx, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
        blue = ms_src.read(blue_band_idx, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')

    # 4. [핵심 수정] 배경의 회색/파란색 노이즈를 0(검은색)으로 지워버리는 유효 마스크 생성
    # 위성 이미지 외곽 노이즈 특성상 특정 밴드가 비정상적으로 높거나 낮은 경우를 타겟팅합니다.
    # 대다수 위성 영상의 진성 검은색 배경 영역이나 경계면 외부 노이즈(임의값)를 필터링합니다.
    invalid_mask = (red <= 1.0) | (green <= 1.0) | (blue <= 1.0) | (pan_data <= 1.0)

    # 5. Brovey 팬샤프닝 연산 수행
    ms_sum = red + green + blue
    ms_sum[ms_sum == 0] = 1e-6

    red_pan = (red / ms_sum) * pan_data
    green_pan = (green / ms_sum) * pan_data
    blue_pan = (blue / ms_sum) * pan_data

    # 6. 마스크를 적용하여 외곽 배경 데이터를 깨끗한 0(검은색)으로 강제 변환
    red_pan[invalid_mask] = 0
    green_pan[invalid_mask] = 0
    blue_pan[invalid_mask] = 0

    # 7. uint16 안전 한계값 범위 제한 및 형변환
    max_val = 65535
    red_pan = np.clip(red_pan, 0, max_val).astype('uint16')
    green_pan = np.clip(green_pan, 0, max_val).astype('uint16')
    blue_pan = np.clip(blue_pan, 0, max_val).astype('uint16')

    # 8. 표준 3밴드 구조로 메타데이터 업데이트 후 파일 저장
    pan_meta.update(count=3, dtype='uint16')
    with rasterio.open(output_file, 'w', **pan_meta) as dest:
        dest.write(red_pan, 1)  # QGIS 표준 1번 밴드(Red)
        dest.write(green_pan, 2)  # QGIS 표준 2번 밴드(Green)
        dest.write(blue_pan, 3)  # QGIS 표준 3번 밴드(Blue)


def process_batch_pipeline(input_dir, output_dir):
    """
    지정된 폴더 내 위성 데이터를 탐색하여 마스킹 기법 기반의 일괄 처리를 수행합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[알림] 결과 저장 폴더 생성 완료: {output_dir}")

    all_files = glob.glob(os.path.join(input_dir, "*.tif"))
    ms_files = [f for f in all_files if "MUL" in os.path.basename(f).upper()]
    pan_files = [f for f in all_files if "PAN" in os.path.basename(f).upper()]

    print(f"[시작] 총 {len(ms_files)}개의 파일 대상 노이즈 정제 파이프라인을 시작합니다.")

    for ms_path in ms_files:
        ms_filename = os.path.basename(ms_path)

        if "MUL" in ms_filename:
            pan_filename = ms_filename.replace("MUL", "PAN")
            out_filename = ms_filename.replace("MUL", "CleanStandardRGB")
        else:
            pan_filename = ms_filename.replace("mul", "pan")
            out_filename = ms_filename.replace("mul", "CleanStandardRGB")

        pan_path = os.path.join(input_dir, pan_filename)
        output_path = os.path.join(output_dir, out_filename)

        if pan_path in pan_files:
            try:
                print(f"\n[처리 중] {ms_filename} 배경 정제 및 융합 중...")
                wv3_masked_pansharpen(ms_path, pan_path, output_path, ms_filename)
                print(f"[완료] 저장 완료 -> {out_filename}")
            except Exception as e:
                print(f"[에러] {ms_filename} 처리 중 오류 발생: {e}")
        else:
            print(f"\n[건너뛰기] {ms_filename}과 일치하는 고해상도 PAN 파일이 없습니다.")

    print("\n[종료] 모든 배치 데이터 정제 파이프라인이 성공적으로 완료되었습니다.")


if __name__ == "__main__":
    source_folder = "wv_data/mul_data"
    result_folder = "wv_data/result"

    process_batch_pipeline(source_folder, result_folder)