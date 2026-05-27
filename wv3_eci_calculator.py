import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from skimage.filters import threshold_otsu
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


def generate_vra_map(input_filepath, growth_stage="early", stress_percentile=15):
    """
    WorldView 다중분광 영상을 기반으로 작물 활력 지수를 산출하고,
    영양 결핍 구역(하위 n%)을 Shapefile로 추출합니다.
    """
    # 1. 파일명 파싱 및 규격화 (FieldCode_Date_Type 형태 적용)
    # 입력 예: SM_01_250728_30_MUL_1.tif
    filename = os.path.basename(input_filepath)
    name_parts = filename.replace('.tif', '').split('_')

    if len(name_parts) >= 3:
        field_code = name_parts[0]  # SM
        date_str = name_parts[2]  # 250728
        # 플랫폼 적재 표준 네이밍 룰 적용
        tif_filename = f"{field_code}_{date_str}_VIGOR.tif"
        shp_filename = f"{field_code}_{date_str}_VRA.shp"
    else:
        raise ValueError(f"파일명 규격 오류: {filename}")

    out_dir = os.path.dirname(input_filepath)
    tif_filepath = os.path.join(out_dir, tif_filename)
    shp_filepath = os.path.join(out_dir, shp_filename)

    print(f"[{field_code} 필지 - {growth_stage} 단계] 영양 결핍 및 활력도 분석 시작...")

    # 2. 위성 영상 데이터 로드
    with rasterio.open(input_filepath) as src:
        meta = src.meta.copy()
        crs = src.crs
        transform = src.transform

        # 밴드 추출 (3: Green, 4: Yellow, 5: Red, 7: NIR1)
        green = src.read(3).astype(np.float32)
        yellow = src.read(4).astype(np.float32)
        red = src.read(5).astype(np.float32)
        nir1 = src.read(7).astype(np.float32)

        # 3. 생육 단계별 마스킹 (토글 로직)
        ndvi = (nir1 - red) / (nir1 + red + 1e-8)

        if growth_stage == "late":
            # 생육 후기: Otsu 알고리즘을 이용해 식생만 엄격하게 분리
            valid_ndvi = ndvi[~np.isnan(ndvi)]
            thresh = threshold_otsu(valid_ndvi)
            plant_mask = ndvi > thresh
        else:
            # 생육 초기: 듬성듬성한 작물과 토양 데이터 포함 분석
            # (NoData 영역인 0 또는 음수 값만 제외)
            plant_mask = (green + yellow) > 0

        # 4. 활력 지수 (Normalized Green-Yellow Index) 산출
        # 공식: (Green - Yellow) / (Green + Yellow)
        # 높을수록 건강(초록), 낮을수록 황화(노랑) 직관과 일치
        vigor_index = np.where(plant_mask, (green - yellow) / (green + yellow + 1e-8), np.nan)

        # 5. TIF 결과물 저장 (플랫폼 시각화용)
        meta.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(tif_filepath, 'w', **meta) as dst:
            dst.write(vigor_index, 1)
        print(f"✅ 활력 지수 맵 생성 완료: {tif_filename}")

        # 6. 처방 지도(VRA) 생성을 위한 임계값 자동 계산
        valid_pixels = vigor_index[~np.isnan(vigor_index)]

        if len(valid_pixels) == 0:
            print("⚠️ 경고: 분석 가능한 유효 픽셀이 없습니다.")
            return None

        # 하위 n%에 해당하는 수치(Threshold) 찾기 (가장 안 좋은 상태)
        threshold_val = np.percentile(valid_pixels, stress_percentile)

        # 7. 결핍 구역(Stress Area) 바이너리 마스크 생성
        stress_mask = np.where(plant_mask & (vigor_index <= threshold_val), 1, 0).astype(np.uint8)

        # 8. 마스크를 폴리곤(Shapefile)으로 벡터화
        geom_results = (
            {'properties': {'class': 'Stress', 'value': v}, 'geometry': s}
            for s, v in shapes(stress_mask, mask=(stress_mask == 1), transform=transform)
        )

        geometries = list(geom_results)

        if geometries:
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.2, preserve_topology=True)
            gdf.to_file(shp_filepath, encoding='utf-8')
            print(f"✅ VRA 처방지도(Shapefile) 추출 완료: {shp_filename}")
        else:
            print("✅ 탐지된 결핍 구역이 없어 Shapefile을 생성하지 않습니다.")

    return tif_filepath, shp_filepath


# ==========================================
# 실행 테스트
# ==========================================
if __name__ == "__main__":
    input_file = "wv_data/mul_data/SM_01_250728_30_MUL_1.tif"
    # 생육 초기 분석 시 growth_stage="early" 설정
    generate_vra_map(input_file, growth_stage="early", stress_percentile=15)