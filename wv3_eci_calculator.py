import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


def generate_early_vra_map(input_filepath, soil_ndvi_threshold=0.2, stress_percentile=15):
    """
    WorldView 8밴드 영상을 기반으로 토양을 제거하고,
    초기 영양 결핍 구역(하위 n%)을 Shapefile로 추출합니다.
    """
    # 1. 파일명 파싱 (플랫폼 네이밍 룰 적용)
    filename = os.path.basename(input_filepath)
    name_parts = filename.replace('.tif', '').split('_')

    if len(name_parts) >= 5:
        field_code, round_num, date_str, resolution = name_parts[0:4]
        # 출력 파일명 설정
        tif_filename = f"{field_code}_{round_num}_{date_str}_{resolution}_VIGOR_1.tif"
        shp_filename = f"{field_code}_{round_num}_{date_str}_{resolution}_VRA_STRESS.shp"
    else:
        raise ValueError(f"파일명 규격 오류: {filename}")

    out_dir = os.path.dirname(input_filepath)
    tif_filepath = os.path.join(out_dir, tif_filename)
    shp_filepath = os.path.join(out_dir, shp_filename)

    print(f"[{field_code} 필지] 토양 마스킹 및 초기 영양 결핍 분석 시작...")

    # 2. 위성 영상 데이터 로드
    with rasterio.open(input_filepath) as src:
        meta = src.meta.copy()
        crs = src.crs
        transform = src.transform

        # 밴드 추출 (1: Coastal, 4: Yellow, 5: Red, 7: NIR1)
        coastal = src.read(1).astype(np.float32)
        yellow = src.read(4).astype(np.float32)
        red = src.read(5).astype(np.float32)
        nir1 = src.read(7).astype(np.float32)

        # 3. 토양 마스킹 (Soil Masking)
        # NDVI 산출 후 threshold 이하(보통 0.2)는 토양으로 간주하여 제외
        ndvi = (nir1 - red) / (nir1 + red + 1e-8)
        plant_mask = ndvi > soil_ndvi_threshold

        # 4. 활력 지수 (Early Vigor Index) 산출 및 역변환
        # ECI 산출
        eci = (yellow - coastal) / (yellow + coastal + 1e-8)

        # 식물 구역에만 (1 - ECI) 적용하여 '높을수록 건강한' 직관적 지수로 변환
        # 토양 구역은 np.nan(결측치)로 처리하여 화면에서 투명하게 뚫리도록 설정
        vigor_index = np.where(plant_mask, 1.0 - eci, np.nan)

        # 5. TIF 결과물 저장 (플랫폼 시각화용)
        meta.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(tif_filepath, 'w', **meta) as dst:
            dst.write(vigor_index, 1)
        print(f"✅ 활력 지수 맵 생성 완료: {tif_filename}")

        # 6. 처방 지도(VRA) 생성을 위한 임계값 자동 계산
        # 결측치(NaN)를 제외한 순수 식물 픽셀 데이터만 추출
        valid_pixels = vigor_index[~np.isnan(vigor_index)]

        if len(valid_pixels) == 0:
            print("⚠️ 경고: 분석 가능한 식물 픽셀이 없습니다. 토양 임계값을 확인하세요.")
            return None

        # 하위 15% (stress_percentile)에 해당하는 수치(Threshold) 찾기
        threshold_val = np.percentile(valid_pixels, stress_percentile)

        # 7. 결핍 구역(Stress Area) 바이너리 마스크 생성
        # 식물 구역(plant_mask)이면서 동시에 활력 지수가 하위 15% 이하인 곳 = 1, 나머지 = 0
        stress_mask = np.where(plant_mask & (vigor_index <= threshold_val), 1, 0).astype(np.uint8)

        # 8. 마스크를 폴리곤(Shapefile)으로 벡터화
        # rasterio.features.shapes를 통해 픽셀 덩어리를 좌표 기반 폴리곤으로 변환
        geom_results = (
            {'properties': {'class': 'Stress', 'value': v}, 'geometry': s}
            for s, v in shapes(stress_mask, mask=(stress_mask == 1), transform=transform)
        )

        geometries = list(geom_results)

        if geometries:
            # GeoDataFrame으로 변환 후 Shapefile 저장
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
            # 폴리곤 경계 단순화 (선택 사항: 노이즈 제거 및 용량 최적화)
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
    generate_early_vra_map(input_file, soil_ndvi_threshold=0.2, stress_percentile=15)
    pass