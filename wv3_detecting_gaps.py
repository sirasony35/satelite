import os
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')


def detect_missing_plants(input_filepath, output_dir, min_gap_area_sqm=0.5):
    """
    30cm 팬샤프닝 영상을 입력받아 MSAVI2를 계산하고,
    일정 면적 이상의 결주(Missing Plant) 구역을 폴리곤으로 추출합니다.

    :param input_filepath: 분석할 위성 영상 파일 경로 (.tif)
    :param output_dir: 결과물을 저장할 폴더 경로
    :param min_gap_area_sqm: 결주로 판정할 최소 면적 (제곱미터). 기본값 0.5㎡
    :return: 추출된 결주 구역 Shapefile(.shp) 경로
    """
    # 1. 파일명 파싱 및 표준 네이밍 룰 적용 (FieldCode_Date_Type)
    filename = os.path.basename(input_filepath)
    name_parts = filename.replace('.tif', '').split('_')

    if len(name_parts) >= 3:
        field_code = name_parts[0]  # SM
        date_str = name_parts[2]  # 250728

        # 출력 파일명 설정 (예: SM_250728_GAPS.shp)
        output_shp = f"{field_code}_{date_str}_GAPS.shp"
    else:
        raise ValueError(f"파일명 규격 오류: {filename}")

    # 2. 결과 저장 폴더가 없으면 자동 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[알림] 결과 저장 폴더 생성 완료: {output_dir}")

    shp_filepath = os.path.join(output_dir, output_shp)

    print(f"[{field_code} 필지 - 결주 구역(Stand Gap) 분석 시작]")

    # 3. 위성 영상 데이터 로드 및 픽셀 면적 계산
    with rasterio.open(input_filepath) as src:
        crs = src.crs
        transform = src.transform

        # 30cm 해상도 기준 1픽셀의 실제 면적 산출 (0.3m * 0.3m = 0.09 sqm)
        pixel_area_sqm = abs(transform[0] * transform[4])
        # 최소 결주 면적을 픽셀 수로 환산
        min_pixels = int(min_gap_area_sqm / pixel_area_sqm)

        # Red(Band 5)와 NIR1(Band 7) 추출 (팬샤프닝 영상 기준)
        red = src.read(5).astype(np.float32)
        nir = src.read(7).astype(np.float32)

        # 4. MSAVI2 지수 산출 (초기 토양 반사율 간섭 최소화)
        radicand = (2 * nir + 1) ** 2 - 8 * (nir - red)
        # 루트 내부가 음수가 되는 노이즈 방지
        radicand = np.where(radicand < 0, 0, radicand)
        msavi2 = (2 * nir + 1 - np.sqrt(radicand)) / 2

        # 5. Otsu 알고리즘으로 식물 vs 흙 이진화
        valid_msavi = msavi2[~np.isnan(msavi2)]

        if len(valid_msavi) == 0:
            print("⚠️ 분석 가능한 유효 픽셀이 없습니다.")
            return None

        threshold = threshold_otsu(valid_msavi)

        # 식물 = 1, 흙(빈 공간) = 0
        plant_mask = msavi2 > threshold

        # 결주 탐지를 위해 흙(빈 공간) 마스크 반전 (빈 공간 = True, 식물 = False)
        gap_mask = (~plant_mask).astype(bool)

        # 6. 형태학적 노이즈 제거 및 크기 필터링 (Blob Analysis)
        filtered_gaps = remove_small_objects(gap_mask, min_size=min_pixels)

        # 7. 바이너리 마스크를 다각형(Shapefile)으로 벡터화
        filtered_gaps_uint = filtered_gaps.astype(np.uint8)

        geom_results = (
            {'properties': {'class': 'Gap', 'area_sqm': v}, 'geometry': s}
            for s, v in shapes(filtered_gaps_uint, mask=(filtered_gaps_uint == 1), transform=transform)
        )

        geometries = list(geom_results)

        # 8. GeoDataFrame 생성 및 저장
        if geometries:
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
            # 폴리곤 단순화 (트랙터 내비게이션 최적화를 위해 경계 노이즈 최소화)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.1, preserve_topology=True)
            gdf.to_file(shp_filepath, encoding='utf-8')
            print(f"✅ 결주 보식용 처방지도 추출 완료: {output_shp}")
            print(f"   - 저장 경로: {shp_filepath}")
            print(f"   - 식별된 결주 구역 개수: {len(gdf)}개")
        else:
            print("✅ 탐지된 결주 구역이 없어 Shapefile을 생성하지 않습니다.")

    return shp_filepath


# ==========================================
# 실행부 (Main Block)
# ==========================================
if __name__ == "__main__":
    # 데이터 경로를 실제 환경에 맞게 지정하세요
    input_file = "wv_data/crop_result/SM01_01_250728_30_VisualLocked.tif"

    # [핵심 변경] 결과물이 저장될 폴더 경로 지정
    result_folder = "wv_data/result_gaps"

    # 함수 실행 (입력 파일, 출력 폴더, 최소 결주 면적 기준 전달)
    try:
        result_shp = detect_missing_plants(input_filepath=input_file, output_dir=result_folder, min_gap_area_sqm=0.5)
    except Exception as e:
        print(f"오류 발생: {e}")