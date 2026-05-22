import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import os
import glob


def get_global_statistics(tif_path):
    """
    [핵심 1] 완전체 위성 영상 전체의 2%~98% 유효 밝기 분포를 추출합니다.
    메모리 초과를 방지하기 위해 영상을 축소(Downsample)하여 고속으로 통계를 냅니다.
    """
    global_stats = []
    with rasterio.open(tif_path) as src:
        # 빠른 통계 추출을 위해 5% 크기로 축소 읽기
        scale_factor = 0.05
        out_shape = (src.count, int(src.height * scale_factor), int(src.width * scale_factor))
        data_downsampled = src.read(out_shape=out_shape).astype('float32')

        for b in range(3):
            band_data = data_downsampled[b]
            # 배경(0) 제외 실제 지표면 픽셀 추출
            valid_pixels = band_data[band_data > 0]

            if len(valid_pixels) > 0:
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                if p98 == p2: p98 = p2 + 1e-6
                global_stats.append((p2, p98))
            else:
                global_stats.append((0, 65535))

    return global_stats


def crop_and_visual_lock(src, shp_path, output_path, global_stats):
    """
    [핵심 2] 글로벌 통계를 사용하여 잘린 영상을 8비트(0~255) RGB로 영구 박제합니다.
    """
    # 1. 벡터 로드 및 마스킹 수행
    gdf = gpd.read_file(shp_path)
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    geometries = gdf.geometry.values
    out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True, nodata=0)

    # 2. 8비트 변환을 위한 빈 배열 생성
    locked_image = np.zeros_like(out_image, dtype='uint8')

    # 3. 원본 완전체의 글로벌 통계(Global Stats)를 기반으로 색상 스케일링
    # 이 과정을 거치면 자른 영상도 원본 완전체와 동일한 렌더링 비율을 가지게 됩니다.
    for b in range(3):
        p2, p98 = global_stats[b]
        band = out_image[b].astype('float32')

        # 글로벌 기준에 맞추어 0~255로 스케일링
        scaled = np.clip(band, p2, p98)
        scaled = (scaled - p2) / (p98 - p2) * 255

        # 배경 마스킹 유지
        scaled[out_image[b] == 0] = 0
        locked_image[b] = scaled.astype('uint8')

    # 4. 메타데이터 업데이트: 8비트(uint8) 및 일반 사진(Photometric=RGB) 인식 강제
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": locked_image.shape[1],
        "width": locked_image.shape[2],
        "transform": out_transform,
        "nodata": 0,
        "dtype": 'uint8',  # 8비트로 변환하여 QGIS 자동 스트레칭 방지
        "photometric": "RGB",  # 일반 사진 색감으로 인식하도록 강제
        "compress": "lzw"
    })

    # 5. 최종 파일 저장
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(locked_image)


def batch_visual_lock_pipeline(tif_dir, shp_dir, output_dir):
    """
    모든 완전체 위성 영상에 대해 색상 박제형 Crop을 일괄 수행합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tif_files = glob.glob(os.path.join(tif_dir, "*_CleanStandardRGB*.tif"))
    shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))

    print(f"[시작] QGIS 왜곡 방지 8비트 시각화 박제 파이프라인 가동")

    for tif_path in tif_files:
        tif_name = os.path.basename(tif_path)
        parts = tif_name.split('_')
        seq_num, date_str, gsd_val = parts[1], parts[2], parts[3]

        print(f"\n>>> 원본 통계 추출 및 분할 중: {tif_name}")

        # 1. 원본 완전체의 글로벌 통계(2~98% 범위) 사전 추출
        global_stats = get_global_statistics(tif_path)

        with rasterio.open(tif_path) as src:
            for shp_path in shp_files:
                field_code = os.path.basename(shp_path).replace(".shp", "")
                out_filename = f"{field_code}_{seq_num}_{date_str}_{gsd_val}_VisualLocked.tif"
                output_path = os.path.join(output_dir, out_filename)

                try:
                    crop_and_visual_lock(src, shp_path, output_path, global_stats)
                    print(f"  [성공] {field_code} 완료 (색감 박제됨)")
                except ValueError:
                    continue
                except Exception as e:
                    print(f"  [에러] {field_code}: {e}")

    print("\n[종료] 모든 필지의 8비트 색상 박제 Crop 작업이 완료되었습니다.")


if __name__ == "__main__":
    input_tif_folder = "wv_data/result"
    input_shp_folder = "wv_data/Shapefile"
    output_crop_folder = "wv_data/crop_result"

    batch_visual_lock_pipeline(input_tif_folder, input_shp_folder, output_crop_folder)