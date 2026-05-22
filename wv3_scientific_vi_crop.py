import rasterio
import rasterio.mask
import geopandas as gpd
import numpy as np
import os
import glob


def crop_scientific_vi(src, shp_path, output_path):
    """
    원본 수치를 100% 보존하며 식생지수(VI) 래스터를 필지별로 정밀하게 자르는 함수.
    배경(No-Data) 처리 시 0.0 값과의 충돌을 방지합니다.
    """
    # 1. 벡터 데이터 로드 및 좌표계 일치화
    gdf = gpd.read_file(shp_path)
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    geometries = gdf.geometry.values

    # 2. 식생지수 전용 NoData 값 설정
    # 원본 파일에 NoData가 정의되어 있다면 그것을 따르고,
    # 없다면 부동소수점(float) 특성에 맞춰 안전한 np.nan (또는 -9999)을 사용합니다.
    # 식생지수에서 0.0은 유효한 값이므로 절대 nodata=0을 사용하면 안 됩니다.
    nodata_val = src.nodata
    if nodata_val is None:
        if src.meta['dtype'].startswith('float'):
            nodata_val = np.nan
        else:
            nodata_val = -9999.0

    # 3. 마스킹 연산 수행 (값 변형 일절 없음)
    out_image, out_transform = rasterio.mask.mask(src, geometries, crop=True, nodata=nodata_val)

    # 4. 출력 메타데이터 설정
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": nodata_val,
        "compress": "lzw"  # 데이터 손실 없는 압축 방식 사용
    })

    # 5. 수치 훼손 없이 있는 그대로 파일 저장
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def batch_scientific_vi_pipeline(tif_dir, shp_dir, output_dir):
    """
    폴더 내의 NDVI, GNDVI, NDRE 등 모든 식생지수 파일을
    12개 필지에 맞게 N:M으로 일괄 자르기 수행합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 특정 지수 이름에 얽매이지 않고, _(지수이름).tif 로 끝나는 모든 VI 파일을 찾습니다.
    # 예: SM_03_260312_30_GNDVI.tif
    tif_files = glob.glob(os.path.join(tif_dir, "*.tif"))
    shp_files = glob.glob(os.path.join(shp_dir, "*.shp"))

    if not tif_files or not shp_files:
        print("[경고] 식생지수 TIF 파일이나 SHP 파일이 지정된 경로에 없습니다.")
        return

    print(f"[시작] 총 {len(tif_files)}개의 식생지수 완전체 영상 과학적 정밀 Crop 시작")

    for tif_path in tif_files:
        tif_name = os.path.basename(tif_path)
        parts = tif_name.replace('.tif', '').split('_')

        # 파일명 파싱 (안전한 추출)
        try:
            # SM_03_260312_30_GNDVI -> parts = ['SM', '03', '260312', '30', 'GNDVI']
            seq_num = parts[1]
            date_str = parts[2]
            gsd_val = parts[3]
            index_type = parts[4]  # GNDVI, NDRE, NDVI 등
        except IndexError:
            print(f"  [건너뛰기] 파일명 규칙이 맞지 않습니다: {tif_name}")
            continue

        print(f"\n>>> 처리 중: {seq_num}회차 {index_type} 지수 ({date_str})")

        with rasterio.open(tif_path) as src:
            for shp_path in shp_files:
                field_code = os.path.basename(shp_path).replace(".shp", "")

                # 출력 파일명 규칙: [필지코드]_[회차]_[날짜]_[GSD]_[지수타입]_Crop.tif
                # 예: SM01_03_260312_30_GNDVI_Crop.tif
                out_filename = f"{field_code}_{seq_num}_{date_str}_{gsd_val}_{index_type}_Crop.tif"
                output_path = os.path.join(output_dir, out_filename)

                try:
                    crop_scientific_vi(src, shp_path, output_path)
                    print(f"  [성공] {field_code} 완료")
                except ValueError:
                    # 해당 필지가 영상 바깥에 있을 경우 조용히 패스
                    pass
                except Exception as e:
                    print(f"  [에러] {field_code} 처리 실패: {e}")

    print("\n[종료] 모든 식생지수 데이터의 수치 보존형 Crop 작업이 완료되었습니다.")


if __name__ == "__main__":
    # 전문가님의 폴더 구조에 맞게 경로를 지정해 주세요.
    # (VI 완전체 파일들이 모여있는 폴더를 input_vi_folder로 지정합니다)
    input_vi_folder = "wv_data/vi_data"  # 식생지수 완전체 파일 폴더
    input_shp_folder = "wv_data/Shapefile"  # 12개 Shapefile 폴더
    output_crop_folder = "wv_data/vi_crop"  # 필지별로 잘린 식생지수 저장 폴더

    batch_scientific_vi_pipeline(input_vi_folder, input_shp_folder, output_crop_folder)