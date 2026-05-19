import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

def wv3_pansharpen(ms_file, pan_file, output_file):
    """
    WorldView-3 다중분광(MS) 영상에서 RGB(5, 3, 2) 밴드를 추출하고,
    팬크로매틱(Pan) 영상과 결합하여 고해상도 RGB 영상을 생성합니다.
    """
    print(f"[{ms_file}] 및 [{pan_file}] 처리 시작...")

    # 1. 팬크로매틱 데이터 읽기 (기준 해상도 및 공간 메타데이터 확보)
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        # 행렬 연산을 위해 데이터를 float32 타입으로 변환하여 읽기
        pan_data = pan_src.read(1).astype('float32')

    # 2. 다중분광 데이터 읽기 및 해상도 일치화(리샘플링)
    with rasterio.open(ms_file) as ms_src:
        # 팬크로매틱 영상의 가로/세로 픽셀 크기에 맞춤
        out_shape = (pan_src.height, pan_src.width)

        # WV3의 RGB 밴드 추출: Red(5), Green(3), Blue(2)
        red = ms_src.read(5, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
        green = ms_src.read(3, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
        blue = ms_src.read(2, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')

    print("데이터 로드 및 리샘플링 완료. 팬샤프닝(Brovey Transform) 적용 중...")

    # 3. Brovey 변환을 이용한 팬샤프닝
    # RGB 밴드의 합계 계산 (0으로 나누는 오류를 방지하기 위해 0인 픽셀은 아주 작은 값 1e-6으로 대체)
    ms_sum = red + green + blue
    ms_sum[ms_sum == 0] = 1e-6

    # 각 밴드의 비율에 팬크로매틱 픽셀 값을 곱하여 고해상도 밴드 생성
    red_pan = (red / ms_sum) * pan_data
    green_pan = (green / ms_sum) * pan_data
    blue_pan = (blue / ms_sum) * pan_data

    # 4. 결과물 저장을 위한 메타데이터 업데이트
    # RGB 3개 밴드로 설정. WV3 영상은 주로 16비트를 사용하므로 uint16 유지
    pan_meta.update(
        count=3,
        dtype='uint16'
    )

    # 5. 파일 저장
    with rasterio.open(output_file, 'w', **pan_meta) as dest:
        dest.write(red_pan.astype('uint16'), 1)
        dest.write(green_pan.astype('uint16'), 2)
        dest.write(blue_pan.astype('uint16'), 3)

    print(f"성공적으로 저장되었습니다: {output_file}")

# ===== 구현 및 실행 부분 =====
if __name__ == "__main__":
    # 입력 파일 경로 설정 (실제 환경의 경로로 수정하여 사용하세요)
    wv3_ms_filepath = "input_WV3_Multispectral.tif"
    wv3_pan_filepath = "input_WV3_Panchromatic.tif"

    # 출력 파일명 설정 (FieldCode_Date_Type 규칙 적용)
    output_filepath = "Gimje01_20260507_PansharpenedRGB.tif"

    if os.path.exists(wv3_ms_filepath) and os.path.exists(wv3_pan_filepath):
        wv3_pansharpen(wv3_ms_filepath, wv3_pan_filepath, output_filepath)
    else:
        print("에러: 입력 파일 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")