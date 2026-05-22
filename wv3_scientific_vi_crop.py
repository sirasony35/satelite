import rasterio
import rasterio.mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import os
import glob


def crop_vi_with_vrt(src, shp_path, output_path):
    """
    WarpedVRT를 사용하여 메모리 상에서 먼저 EPSG:5179로 재투영한 후,
    5179 좌표계의 폴리곤으로 자름으로써 외곽선 테두리 왜곡을 원천 차단합니다.
    """
    dst_crs = 'EPSG:5179'

    # 1. 벡터 데이터 로드 및 5179 좌표계로 변환 (자르는 칼날을 5179로 맞춤)
    gdf = gpd.read_file(shp_path)
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)

    geometries = gdf.geometry.values

    # 2. 식생지수 전용 NoData 값 설정 (원본 보존)
    nodata_val = src.nodata
    if nodata_val is None:
        if src.meta['dtype'].startswith('float'):
            nodata_val = np.nan
        else:
            nodata_val = -9999.0

    # 3. [핵심] 원본 래스터를 메모리 상에서 5179로 가상 투영 (WarpedVRT)
    # 식생지수 수치 보존을 위해 Bilinear 사용
    with WarpedVRT(src, crs=dst_crs, resampling=Resampling.bilinear) as vrt:

        # 4. 5179로 정렬된 가상 래스터 위에서, 5179 폴리곤 모양으로 깔끔하게 자르기
        # 이 순서로 진행해야 회전으로 인한 외곽 NoData 테두리 현상이 발생하지 않습니다.
        out_image, out_transform = rasterio.mask.mask(vrt, geometries, crop=True, nodata=nodata_val)

        # 5. 출력 메타데이터 설정
        out_meta = vrt.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": nodata_val,
            "compress": "lzw"
        })

        # 6. 파일 저장
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def batch_scientific_vi_pipeline(tif_dir, shp_dir, output_dir):
    """
    식생지수 TIF 파일들을 5179 좌표계 기반으로 테두리 없이 깔끔하게 분할합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tif_files = glob.glob(os.path.join(tif_dir, "*.tif"))
    shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))

    if not tif_files or not shp_files:
        print("[경고] 식생지수 TIF 또는 SHP 파일이 없습니다.")
        return

    print(f"[시작] 총 {len(tif_files)}개의 식생지수 대상 [가상 투영(VRT) 기반 정밀 Crop] 시작")

    for tif_path in tif_files:
        tif_name = os.path.basename(tif_path)
        parts = tif_name.replace('.tif', '').split('_')

        try:
            seq_num, date_str, gsd_val, index_type = parts[1], parts[2], parts[3], parts[4]
        except IndexError:
            continue

        print(f"\n>>> 처리 중: {seq_num}회차 {index_type} 지수 ({date_str})")

        with rasterio.open(tif_path) as src:
            for shp_path in shp_files:
                field_code = os.path.basename(shp_path).replace(".shp", "")

                out_filename = f"{field_code}_{seq_num}_{date_str}_{gsd_val}_{index_type}_Crop.tif"
                output_path = os.path.join(output_dir, out_filename)

                try:
                    crop_vi_with_vrt(src, shp_path, output_path)
                    print(f"  [성공] {field_code} 완료 (EPSG:5179 테두리 없음)")
                except ValueError:
                    pass
                except Exception as e:
                    print(f"  [에러] {field_code}: {e}")

    print("\n[종료] 식생지수 데이터의 외곽선 왜곡 없는 5179 Crop 작업이 완료되었습니다.")


if __name__ == "__main__":
    input_vi_folder = "wv_data/vi_data"
    input_shp_folder = "wv_data/Shapefile"
    output_crop_folder = "wv_data/vi_crop"

    batch_scientific_vi_pipeline(input_vi_folder, input_shp_folder, output_crop_folder)