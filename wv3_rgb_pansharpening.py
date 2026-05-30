import rasterio
from rasterio.enums import Resampling, ColorInterp
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
import numpy as np
import os
import glob


def wv3_pansharpen_dual_export_5179(ms_file, pan_file, out_8b_file, out_rgb_file):
    """
    QGIS GDAL 알고리즘으로 8밴드 팬샤프닝 및 5179 변환을 수행한 후,
    1. 분석용 8밴드 데이터 (16-bit, 원본 수치 보존)
    2. 시각화용 3밴드 RGB 데이터 (8-bit, NoData 충돌 방지 및 밸런스 평탄화 적용)
    두 가지 결과물을 동시에 출력합니다.
    """
    # 1. 팬크로매틱 데이터 로드
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        pan_data = pan_src.read(1).astype('float32')
        out_shape = (pan_src.height, pan_src.width)

        src_crs = pan_src.crs
        src_transform = pan_src.transform
        src_bounds = pan_src.bounds
        src_width = pan_src.width
        src_height = pan_src.height

    # 2. MS 8밴드 전체 로드 (Bilinear 적용)
    ms_bands = []
    with rasterio.open(ms_file) as ms_src:
        for i in range(1, 9):
            band_data = ms_src.read(i, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
            ms_bands.append(band_data)

    ms_stack = np.stack(ms_bands)

    # 0으로 나누기 방지
    ms_stack[ms_stack == 0] = 1e-6
    pan_data[pan_data == 0] = 1e-6

    # 3. QGIS(GDAL) 기본 팬샤프닝 알고리즘 (8밴드 평균 Pseudo-Pan)
    ms_mean = np.mean(ms_stack, axis=0)
    ms_mean[ms_mean == 0] = 1e-6

    ratio = pan_data / ms_mean
    sharpened_stack = ms_stack * ratio

    # 4. uint16 안전 클리핑
    sharpened_stack = np.clip(sharpened_stack, 0, 65535).astype('uint16')

    # ---------------------------------------------------------
    # 5. EPSG:5179 좌표계 재투영 (Bilinear)
    # ---------------------------------------------------------
    dst_crs = 'EPSG:5179'
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds
    )

    reprojected_stack = np.zeros((8, dst_height, dst_width), dtype='uint16')

    reproject(
        source=sharpened_stack,
        destination=reprojected_stack,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=WarpResampling.bilinear,
        nodata=0
    )

    # ---------------------------------------------------------
    # 6. 결과물 #1: 분석용 8밴드 원본 수치 저장 (16-bit)
    # ---------------------------------------------------------
    pan_meta.update({
        "count": 8,
        "dtype": 'uint16',
        "crs": dst_crs,
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "nodata": 0,
        "compress": "lzw"
    })

    with rasterio.open(out_8b_file, 'w', **pan_meta) as dest_8b:
        dest_8b.write(reprojected_stack)

    # ---------------------------------------------------------
    # 7. 결과물 #2: 시각화용 RGB 슬라이싱 및 빵꾸(NoData) 방지 보정 (8-bit)
    # ---------------------------------------------------------
    # WV3 스펙 기준: Band 5(Red) -> idx 4, Band 3(Green) -> idx 2, Band 2(Blue) -> idx 1
    r_idx, g_idx, b_idx = 4, 2, 1

    rgb_raw = reprojected_stack[[r_idx, g_idx, b_idx], :, :].astype('float32')
    rgb_normalized = np.zeros_like(rgb_raw, dtype='uint8')

    # [핵심] 진짜 외곽 투명 배경과 내부의 어두운 픽셀을 수학적으로 구분하기 위한 마스크
    bg_mask = (rgb_raw[0] == 0) & (rgb_raw[1] == 0) & (rgb_raw[2] == 0)

    for i in range(3):
        band = rgb_raw[i]
        valid_pixels = band[~bg_mask]  # 진짜 외곽 배경을 제외한 내부 픽셀만 추출

        if len(valid_pixels) > 0:
            p2, p98 = np.percentile(valid_pixels, (2, 98))
            if p98 == p2: p98 = p2 + 1e-6

            # [해결] 0~255가 아닌 1~255 스케일로 강제 맵핑합니다.
            # 내부의 짙은 그림자나 검은색 픽셀이 1이 되어, QGIS에서 투명하게 뚫리는 현상을 차단합니다.
            stretched = np.clip(band, p2, p98)
            stretched = (stretched - p2) / (p98 - p2) * 254 + 1

            # 진짜 외곽 배경만 다시 0(투명)으로 돌려놓습니다.
            stretched[bg_mask] = 0
            rgb_normalized[i] = stretched.astype('uint8')

    # 시각화용 메타데이터 갱신
    pan_meta.update({
        "count": 3,
        "dtype": 'uint8',
        "photometric": "RGB",
        "nodata": 0  # 0은 오직 '외곽 배경'만을 의미하게 됨
    })

    with rasterio.open(out_rgb_file, 'w', **pan_meta) as dest_rgb:
        dest_rgb.write(rgb_normalized)
        dest_rgb.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]


def process_batch_pipeline(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = glob.glob(os.path.join(input_dir, "*.tif"))
    ms_files = [f for f in all_files if "MUL" in os.path.basename(f).upper()]
    pan_files = [f for f in all_files if "PAN" in os.path.basename(f).upper()]

    for ms_path in ms_files:
        ms_filename = os.path.basename(ms_path)

        if "MUL" in ms_filename:
            pan_filename = ms_filename.replace("MUL", "PAN")
            out_8b_name = ms_filename.replace("MUL", "PANSHARP_8B")
            out_rgb_name = ms_filename.replace("MUL", "CleanStandardRGB")
        else:
            pan_filename = ms_filename.replace("mul", "pan")
            out_8b_name = ms_filename.replace("mul", "PANSHARP_8B")
            out_rgb_name = ms_filename.replace("mul", "CleanStandardRGB")

        pan_path = os.path.join(input_dir, pan_filename)
        out_8b_path = os.path.join(output_dir, out_8b_name)
        out_rgb_path = os.path.join(output_dir, out_rgb_name)

        if pan_path in pan_files:
            try:
                print(f"\n[처리 중] {ms_filename} -> 듀얼 출력 및 NoData 방지 처리 중")
                wv3_pansharpen_dual_export_5179(ms_path, pan_path, out_8b_path, out_rgb_path)
                print(f"  [성공 1] {out_8b_name} 저장 완료")
                print(f"  [성공 2] {out_rgb_name} 저장 완료")
            except Exception as e:
                print(f"  [에러] 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    source_folder = "wv_data/mul_data"
    result_folder = "wv_data/result_pan"
    process_batch_pipeline(source_folder, result_folder)