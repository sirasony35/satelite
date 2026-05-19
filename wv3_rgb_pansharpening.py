import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
import glob


def wv3_pansharpen(ms_file, pan_file, output_file):
    """
    안정화된 연산이 적용된 WorldView-3 팬샤프닝 함수
    """
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        pan_data = pan_src.read(1).astype('float32')

    with rasterio.open(ms_file) as ms_src:
        out_shape = (pan_src.height, pan_src.width)

        # 5, 3, 2 밴드 로드 및 리샘플링
        red = ms_src.read(5, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
        green = ms_src.read(3, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
        blue = ms_src.read(2, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')

    # [수정] 0 나누기 오류 및 극단적 분모 값 제어 방지 안정화
    ms_sum = red + green + blue
    ms_sum[ms_sum == 0] = 1e-6

    # Brovey 변환 수행
    red_pan = (red / ms_sum) * pan_data
    green_pan = (green / ms_sum) * pan_data
    blue_pan = (blue / ms_sum) * pan_data

    # [추가] 데이터 오버플로우 방지를 위한 가이드라인 (uint16 범위 내 클리핑)
    # WV3의 유효 최대 반사값(일반적으로 DN 2047 또는 대기보정 시 다른 기준)에 맞춰 상한선 제한
    # 여기서는 안전하게 16비트 최대값(65535) 내로 바인딩하되, 필요시 실제 영상 데이터의 최대값으로 조절 가능합니다.
    max_val = 65535
    red_pan = np.clip(red_pan, 0, max_val)
    green_pan = np.clip(green_pan, 0, max_val)
    blue_pan = np.clip(blue_pan, 0, max_val)

    # 메타데이터 업데이트 및 저장
    pan_meta.update(count=3, dtype='uint16')

    with rasterio.open(output_file, 'w', **pan_meta) as dest:
        dest.write(red_pan.astype('uint16'), 1)
        dest.write(green_pan.astype('uint16'), 2)
        dest.write(blue_pan.astype('uint16'), 3)


def process_flexible_batch(input_dir, output_dir):
    """
    파일명에 'MUL'과 'PAN' 키워드가 포함된 파일들을 찾아 유연하게 매칭하고 일괄 처리합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[알림] 결과 저장 폴더 생성 완료: {output_dir}")

    # 1. 입력 폴더 내의 모든 TIF 파일 탐색
    all_files = glob.glob(os.path.join(input_dir, "*.tif"))

    # 2. 키워드 기반 분리 (대소문자 구분 없이 처리하기 위해 .upper() 사용)
    ms_files = [f for f in all_files if "MUL" in os.path.basename(f).upper()]
    pan_files = [f for f in all_files if "PAN" in os.path.basename(f).upper()]

    if not ms_files:
        print(f"[경고] {input_dir} 폴더 내에 'MUL' 키워드가 포함된 TIF 파일이 없습니다.")
        return

    print(f"[시작] 총 {len(ms_files)}개의 멀티스펙트럴 파일 매칭을 시작합니다.")

    # 3. 매칭 및 루프 실행
    for ms_path in ms_files:
        ms_filename = os.path.basename(ms_path)

        # 'MUL'을 'PAN'으로 변환하여 매칭할 팬크로매틱 파일명 예측
        # 대소문자 변형에 대응하기 위해 'MUL'과 'mul' 모두 대응하도록 처리
        if "MUL" in ms_filename:
            pan_filename = ms_filename.replace("MUL", "PAN")
            out_filename = ms_filename.replace("MUL", "PansharpenedRGB")
        else:
            pan_filename = ms_filename.replace("mul", "pan")
            out_filename = ms_filename.replace("mul", "PansharpenedRGB")

        pan_path = os.path.join(input_dir, pan_filename)
        output_path = os.path.join(output_dir, out_filename)

        # 4. 동적 매칭 검증 후 팬샤프닝 실행
        if pan_path in pan_files:
            try:
                print(f"\n[처리 중] {ms_filename} <-> {pan_filename}")
                wv3_pansharpen(ms_path, pan_path, output_path)
                print(f"[완료] 저장 완료 -> {out_filename}")
            except Exception as e:
                print(f"[에러] {ms_filename} 처리 중 오류 발생: {e}")
        else:
            print(f"\n[건너뛰기] {ms_filename}과 매칭되는 팬크로매틱 파일을 찾을 수 없습니다. (예상 파일명: {pan_filename})")

    print("\n[종료] 모든 배치 처리가 완료되었습니다.")


# ===== 실행 블록 =====
if __name__ == "__main__":
    source_folder = "wv_data/mul_data"
    result_folder = "wv_data/result"

    process_flexible_batch(source_folder, result_folder)