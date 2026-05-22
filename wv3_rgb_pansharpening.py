import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
import numpy as np
import os
import glob


def wv3_masked_pansharpen_5179(ms_file, pan_file, output_file, filename_context):
    """
    배경 노이즈 마스킹, 동적 밴드 정렬 및 Cubic 융합을 처리한 뒤,
    최종 결과물을 대한민국 국가 표준 좌표계(EPSG:5179)로 즉시 재투영하는 통합 함수
    """
    # 1. 팬크로매틱 고해상도 메타데이터 및 밝기값 로드
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        pan_data = pan_src.read(1).astype('float32')
        out_shape = (pan_src.height, pan_src.width)

        # 재투영(Reproject) 연산을 위해 원본의 공간 정보 추출
        src_crs = pan_src.crs
        src_transform = pan_src.transform
        src_bounds = pan_src.bounds
        src_width = pan_src.width
        src_height = pan_src.height

    # 2. 파일명 컨텍스트 분석 (SM_07 밴드 뒤틀림 예외 처리)
    if "SM_07" in filename_context.upper():
        red_band_idx, green_band_idx, blue_band_idx = 5, 2, 3
        print(f"-> [배경정제 및 밴드교정] {filename_context}: R=5, G=2, B=3 적용")
    else:
        red_band_idx, green_band_idx, blue_band_idx = 5, 3, 2

    # 3. 정해진 인덱스에 따라 3차 회선(Cubic) 리샘플링 데이터 로드
    with rasterio.open(ms_file) as ms_src:
        red = ms_src.read(red_band_idx, out_shape=out_shape, resampling=Resampling.cubic).astype('float32')
        green = ms_src.read(green_band_idx, out_shape=out_shape, resampling=Resampling.cubic).astype('float32')
        blue = ms_src.read(blue_band_idx, out_shape=out_shape, resampling=Resampling.cubic).astype('float32')

    # 4. 배경 노이즈(블루/회색) 마스크 생성
    invalid_mask = (red <= 1.0) | (green <= 1.0) | (blue <= 1.0) | (pan_data <= 1.0)

    # 5. Brovey 팬샤프닝 융합 연산
    ms_sum = red + green + blue
    ms_sum[ms_sum == 0] = 1e-6

    red_pan = (red / ms_sum) * pan_data
    green_pan = (green / ms_sum) * pan_data
    blue_pan = (blue / ms_sum) * pan_data

    # 6. 외곽 배경 데이터를 깨끗한 0(검은색)으로 강제 변환
    red_pan[invalid_mask] = 0
    green_pan[invalid_mask] = 0
    blue_pan[invalid_mask] = 0

    # 7. uint16 안전 클리핑 및 배열 병합 (Reproject를 위한 Stack 준비)
    max_val = 65535
    red_pan = np.clip(red_pan, 0, max_val).astype('uint16')
    green_pan = np.clip(green_pan, 0, max_val).astype('uint16')
    blue_pan = np.clip(blue_pan, 0, max_val).astype('uint16')

    # 3개의 밴드를 하나의 3D 배열(Stack)로 합칩니다.
    sharpened_stack = np.stack([red_pan, green_pan, blue_pan])

    # ---------------------------------------------------------
    # 8. [신규 추가] EPSG:5179 좌표계 재투영(Reprojection) 블록
    # ---------------------------------------------------------
    dst_crs = 'EPSG:5179'

    # 목적지 좌표계에 맞는 새로운 해상도 그리드와 변환 행렬(Transform) 계산
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds
    )

    # 투영된 이미지를 담을 빈 배열 생성
    reprojected_stack = np.zeros((3, dst_height, dst_width), dtype='uint16')

    # 공간 워핑(Warping) 가동
    # ※ 주의: 대용량 완전체 연산이므로 컴퓨터 사양에 따라 다소 시간이 소요될 수 있습니다.
    reproject(
        source=sharpened_stack,
        destination=reprojected_stack,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=WarpResampling.cubic,  # 시각화 품질 보존을 위한 Cubic 필터
        nodata=0
    )

    # 9. 메타데이터 스펙 업데이트 및 저장
    pan_meta.update({
        "count": 3,
        "dtype": 'uint16',
        "crs": dst_crs,  # EPSG:5179 스펙 확정
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "nodata": 0,
        "compress": "lzw"  # 용량 최적화
    })

    with rasterio.open(output_file, 'w', **pan_meta) as dest:
        dest.write(reprojected_stack)


def process_batch_pipeline(input_dir, output_dir):
    """
    폴더 내 데이터를 탐색하여 일괄 팬샤프닝 및 5179 재투영 처리를 수행합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[알림] 결과 저장 폴더 생성 완료: {output_dir}")

    all_files = glob.glob(os.path.join(input_dir, "*.tif"))
    ms_files = [f for f in all_files if "MUL" in os.path.basename(f).upper()]
    pan_files = [f for f in all_files if "PAN" in os.path.basename(f).upper()]

    print(f"[시작] 총 {len(ms_files)}개 파일 대상 [팬샤프닝 + EPSG:5179 변환] 가동")

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
                print(f"\n[처리 중] {ms_filename} -> 융합 및 EPSG:5179 변환 중...")
                wv3_masked_pansharpen_5179(ms_path, pan_path, output_path, ms_filename)
                print(f"[완료] 저장 완료 -> {out_filename}")
            except Exception as e:
                print(f"[에러] {ms_filename} 처리 중 오류 발생: {e}")
        else:
            print(f"\n[건너뛰기] 일치하는 고해상도 PAN 파일이 없습니다.")

    print("\n[종료] 전체 데이터의 팬샤프닝 및 5179 좌표 변환이 완료되었습니다.")


if __name__ == "__main__":
    # 데이터 경로를 실제 환경에 맞게 지정하세요
    source_folder = "wv_data/mul_data"
    result_folder = "wv_data/resul_pan"

    process_batch_pipeline(source_folder, result_folder)