import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
import numpy as np
import os
import glob


def wv3_masked_pansharpen_5179(ms_file, pan_file, output_file, filename_context):
    """
    (초기 원본 로직 적용) Brovey 팬샤프닝을 8개 밴드로 확장하고,
    최종 결과물을 EPSG:5179 좌표계의 16-bit 8밴드 TIF로 저장합니다.
    """
    # 1. 팬크로매틱 고해상도 메타데이터 및 밝기값 로드
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        pan_data = pan_src.read(1).astype('float32')
        out_shape = (pan_src.height, pan_src.width)

        src_crs = pan_src.crs
        src_transform = pan_src.transform
        src_bounds = pan_src.bounds
        src_width = pan_src.width
        src_height = pan_src.height

    # 2. 마스킹을 위한 RGB 인덱스 설정 (0-based index)
    if "SM_07" in filename_context.upper():
        r_idx, g_idx, b_idx = 4, 1, 2
        print(f"-> [배경정제 및 밴드교정] {filename_context}: R=5, G=2, B=3 기준 적용")
    else:
        r_idx, g_idx, b_idx = 4, 2, 1

    # 3. 8개 밴드 전체 로드 (원본 코드의 Cubic 리샘플링 유지)
    ms_bands = []
    with rasterio.open(ms_file) as ms_src:
        for i in range(1, 9):
            band_data = ms_src.read(i, out_shape=out_shape, resampling=Resampling.cubic).astype('float32')
            ms_bands.append(band_data)

    # 3D 배열(Stack)로 결합 (형태: 8, Height, Width)
    ms_stack = np.stack(ms_bands)

    # 4. 배경 노이즈(블루/회색) 마스크 생성 (원본 로직 동일 적용)
    invalid_mask = (ms_stack[r_idx] <= 1.0) | (ms_stack[g_idx] <= 1.0) | (ms_stack[b_idx] <= 1.0) | (pan_data <= 1.0)

    # 5. Brovey 팬샤프닝 융합 연산 (8밴드 확장)
    # 기존: R + G + B -> 변경: 8개 밴드 전체의 합
    ms_sum = np.sum(ms_stack, axis=0)
    ms_sum[ms_sum == 0] = 1e-6

    # 공식: (각 밴드 / 전체 합) * Pan
    # 넘파이 브로드캐스팅을 통해 8개 밴드에 일괄 연산 적용
    ratio = pan_data / ms_sum
    sharpened_stack = ms_stack * ratio

    # 6. 외곽 배경 데이터를 깨끗한 0(검은색)으로 강제 변환
    sharpened_stack[:, invalid_mask] = 0

    # 7. uint16 안전 클리핑 (원본 로직 유지)
    max_val = 65535
    sharpened_stack = np.clip(sharpened_stack, 0, max_val).astype('uint16')

    # ---------------------------------------------------------
    # 8. EPSG:5179 좌표계 재투영(Reprojection) 블록
    # ---------------------------------------------------------
    dst_crs = 'EPSG:5179'

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds
    )

    # 8밴드를 담을 빈 배열 생성
    reprojected_stack = np.zeros((8, dst_height, dst_width), dtype='uint16')

    reproject(
        source=sharpened_stack,
        destination=reprojected_stack,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=WarpResampling.cubic,
        nodata=0
    )

    # 9. 메타데이터 스펙 업데이트 및 저장 (8밴드)
    pan_meta.update({
        "count": 8,  # 8개 밴드로 변경
        "dtype": 'uint16',
        "crs": dst_crs,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "nodata": 0,
        "compress": "lzw"
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

    print(f"[시작] 총 {len(ms_files)}개 파일 대상 [8밴드 팬샤프닝 + EPSG:5179 변환] 가동")

    for ms_path in ms_files:
        ms_filename = os.path.basename(ms_path)

        if "MUL" in ms_filename:
            pan_filename = ms_filename.replace("MUL", "PAN")
            # 출력 파일명 변경 (명확성을 위해 PANSHARP_8B 적용)
            out_filename = ms_filename.replace("MUL", "PANSHARP_8B")
        else:
            pan_filename = ms_filename.replace("mul", "pan")
            out_filename = ms_filename.replace("mul", "PANSHARP_8B")

        pan_path = os.path.join(input_dir, pan_filename)
        output_path = os.path.join(output_dir, out_filename)

        if pan_path in pan_files:
            try:
                print(f"\n[처리 중] {ms_filename} -> 8밴드 융합 및 변환 중...")
                wv3_masked_pansharpen_5179(ms_path, pan_path, output_path, ms_filename)
                print(f"[완료] 저장 완료 -> {out_filename}")
            except Exception as e:
                print(f"[에러] {ms_filename} 처리 중 오류 발생: {e}")
        else:
            print(f"\n[건너뛰기] 일치하는 고해상도 PAN 파일이 없습니다.")

    print("\n[종료] 전체 데이터의 8밴드 팬샤프닝 및 5179 좌표 변환이 완료되었습니다.")


if __name__ == "__main__":
    # 데이터 경로를 실제 환경에 맞게 지정하세요
    source_folder = "wv_data/mul_data"
    result_folder = "wv_data/result"

    process_batch_pipeline(source_folder, result_folder)