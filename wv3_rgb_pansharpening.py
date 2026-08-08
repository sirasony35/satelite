import rasterio
from rasterio.enums import Resampling, ColorInterp
from rasterio.warp import calculate_default_transform, reproject, Resampling as WarpResampling
import numpy as np
import os
import glob


# WV3 PAN(450-800nm) 응답대역에 맞춘 Pseudo-PAN 가중치
# Coastal(B1, 400-450nm)과 NIR2(B8, 860-1040nm)는 PAN 응답 밖 → 기여 0
# 나머지 5개 가시·근적외 밴드는 PAN 대역 내에서 합이 1이 되도록 정규화
PAN_WEIGHTS_WV3 = np.array(
    [0.00, 0.18, 0.21, 0.16, 0.21, 0.13, 0.11, 0.00],
    dtype='float32'
)

# WorldView Legion 8밴드 (Coastal/Blue/Green/Yellow/Red/RedEdge1/RedEdge2/NIR)용 가중치
# WV3와 달리 B6/B7이 RedEdge1/RedEdge2이고 B8이 유일한 NIR.
# PAN(450-800nm) 대역 내 각 밴드의 유효대역폭(XML EFFECTIVEBANDWIDTH) 비례 배분:
#   Coastal(400-450) 밖 → 0 / NIR(770-895)은 770-800 구간만 부분 기여
PAN_WEIGHTS_LEGION = np.array(
    [0.00, 0.20, 0.23, 0.13, 0.20, 0.07, 0.07, 0.10],
    dtype='float32'
)


def _box_blur(arr, ksize=5):
    """
    분리형 cumsum을 이용한 박스 평활 (scipy 없이 numpy만으로 저주파 추출).
    경계는 reflect 패딩으로 처리. 가우시안과 유사한 저주파 응답.
    """
    arr_f = arr.astype(np.float64)
    pad = ksize // 2
    padded = np.pad(arr_f, pad, mode='reflect')

    # 가로 박스: cumsum + 0 prepend 후 차분
    cs = np.cumsum(padded, axis=1)
    cs = np.column_stack([np.zeros((cs.shape[0], 1)), cs])
    h_blurred = (cs[:, ksize:] - cs[:, :-ksize]) / ksize

    # 세로 박스
    cs = np.cumsum(h_blurred, axis=0)
    cs = np.vstack([np.zeros((1, cs.shape[1])), cs])
    v_blurred = (cs[ksize:, :] - cs[:-ksize, :]) / ksize

    return v_blurred.astype(arr.dtype)


def wv3_pansharpen_dual_export_5179(ms_file, pan_file, out_8b_file, out_rgb_file,
                                    pan_weights=None):
    """
    PAN 응답대역 가중 Pseudo-PAN으로 팬샤프닝 후 EPSG:5179로 재투영.
    1. 분석용 8밴드 (uint16, 원본 수치 보존)
    2. 시각화용 RGB (uint8, 유효 footprint 한정 스트레칭으로 타일간 색감 안정)

    :param pan_weights: 밴드별 Pseudo-PAN 가중치 (8,). None이면 Legion 가중치.
                        WV3 원본 처리 시 PAN_WEIGHTS_WV3 명시 지정.
    """
    if pan_weights is None:
        pan_weights = PAN_WEIGHTS_LEGION
    # 1. PAN 로드 (원본 NoData 정보 유지)
    with rasterio.open(pan_file) as pan_src:
        pan_meta = pan_src.meta.copy()
        pan_data = pan_src.read(1).astype('float32')
        out_shape = (pan_src.height, pan_src.width)

        src_crs = pan_src.crs
        src_transform = pan_src.transform
        src_bounds = pan_src.bounds
        src_width = pan_src.width
        src_height = pan_src.height
        pan_nodata = pan_src.nodata if pan_src.nodata is not None else 0

    # 2. MS 8밴드 로드 (PAN 해상도로 Bilinear 업샘플)
    ms_bands = []
    with rasterio.open(ms_file) as ms_src:
        ms_nodata = ms_src.nodata if ms_src.nodata is not None else 0
        for i in range(1, 9):
            band_data = ms_src.read(i, out_shape=out_shape, resampling=Resampling.bilinear).astype('float32')
            ms_bands.append(band_data)

    ms_stack = np.stack(ms_bands)  # (8, H, W)

    # 3. 유효 footprint 마스크 (PAN과 MUL 모두 유효한 픽셀)
    # 한쪽만 NoData인 경계 픽셀에서 ratio 폭주를 막아 outlier 발생을 차단
    pan_valid = pan_data != pan_nodata
    ms_valid = np.all(ms_stack != ms_nodata, axis=0)
    valid_mask = pan_valid & ms_valid  # (H, W) bool

    # 4. 가중 Pseudo-PAN 산출 후 ratio 계산
    weights = pan_weights[:, None, None]
    pseudo_pan = np.sum(ms_stack * weights, axis=0)

    # 0 분모 방지: 유효 영역 외부는 어차피 마스킹되므로 안전한 값으로 채움
    safe_pseudo = np.where(pseudo_pan > 1e-3, pseudo_pan, 1.0)
    ratio = np.where(valid_mask, pan_data / safe_pseudo, 0.0)

    sharpened_stack = ms_stack * ratio[None, :, :]
    # NoData 전파: 유효 영역 밖은 강제 0으로 깎아 outlier가 통계에 끼지 못하게 함
    sharpened_stack = np.where(valid_mask[None, :, :], sharpened_stack, 0.0)
    sharpened_stack = np.clip(sharpened_stack, 0, 65535).astype('uint16')

    # 5. EPSG:5179 재투영 (Bilinear). valid_mask는 Nearest로 별도 전파
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
        src_nodata=0,
        dst_nodata=0,
    )

    # 유효 footprint를 재투영해 시각화 스트레칭의 통계 마스크로 사용
    valid_reproj = np.zeros((dst_height, dst_width), dtype='uint8')
    reproject(
        source=valid_mask.astype('uint8'),
        destination=valid_reproj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=WarpResampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    valid_reproj = valid_reproj.astype(bool)

    # 6. 결과물 #1: 분석용 8밴드 원본 수치 저장 (16-bit)
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

    # 7. 결과물 #2: 시각화용 RGB (8-bit) — HPF 디테일 주입 방식으로 합성
    # IHS/Brovey의 곱셈 ratio는 식생 밀도 높은 타일에서 PAN/visible 분포 차이로
    # 픽셀별 ratio가 폭주해 색감을 왜곡함. HPF는 PAN의 zero-mean 고주파 성분을
    # 가산하므로 DC(평균 색상)가 수학적으로 보존됨 → QGIS 색감과 일치
    #
    # 절차:
    #   (a) PAN을 가우시안 저주파로 평활화 → pan_low
    #   (b) pan_detail = PAN - pan_low (mean ≈ 0인 고주파 성분)
    #   (c) gain_X = std(src_X) / std(pan_detail) — 밴드별 자연 분산에 정합
    #   (d) new_X = src_X + gain_X × pan_detail — DC 보존, HF만 가산

    src_R = ms_stack[4]  # Band 5: Red
    src_G = ms_stack[2]  # Band 3: Green
    src_B = ms_stack[1]  # Band 2: Blue

    # PAN의 고주파 성분 분리 (5x5 박스 평활 — MUL/PAN 4배 업샘플 폭에 대응)
    pan_low = _box_blur(pan_data, ksize=5)
    pan_detail = pan_data - pan_low

    # 밴드별 게인 산출 (PAN 디테일 강도를 각 밴드 분산에 맞춤)
    pd_std = pan_detail[valid_mask].std() if valid_mask.any() else 0.0
    if pd_std > 1e-6:
        gain_R = min(src_R[valid_mask].std() / pd_std, 2.0)
        gain_G = min(src_G[valid_mask].std() / pd_std, 2.0)
        gain_B = min(src_B[valid_mask].std() / pd_std, 2.0)
    else:
        gain_R = gain_G = gain_B = 0.0

    # 가산 주입 — DC 보존, HF만 가산
    new_R = src_R + gain_R * pan_detail
    new_G = src_G + gain_G * pan_detail
    new_B = src_B + gain_B * pan_detail

    hpf_rgb_src = np.stack([new_R, new_G, new_B])
    hpf_rgb_src = np.where(valid_mask[None, :, :], hpf_rgb_src, 0.0)
    hpf_rgb_src = np.clip(hpf_rgb_src, 0, 65535).astype('uint16')

    # EPSG:5179 재투영 (RGB 전용)
    hpf_rgb_reproj = np.zeros((3, dst_height, dst_width), dtype='uint16')
    reproject(
        source=hpf_rgb_src,
        destination=hpf_rgb_reproj,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=WarpResampling.bilinear,
        src_nodata=0,
        dst_nodata=0,
    )

    # 밴드별 p2/p98 스트레칭 — HPF가 DC를 보존하므로 QGIS와 동일한 결과
    rgb_normalized = np.zeros((3, dst_height, dst_width), dtype='uint8')
    for i in range(3):
        band = hpf_rgb_reproj[i].astype('float32')
        valid_pix = band[valid_reproj & (band > 0)]
        if len(valid_pix) > 0:
            p2, p98 = np.percentile(valid_pix, (2, 98))
            if p98 - p2 < 1e-6:
                p98 = p2 + 1.0
            stretched = np.clip(band, p2, p98)
            stretched = (stretched - p2) / (p98 - p2) * 254.0 + 1.0
            stretched[~valid_reproj] = 0
            rgb_normalized[i] = stretched.astype('uint8')

    pan_meta.update({
        "count": 3,
        "dtype": 'uint8',
        "photometric": "RGB",
        "nodata": 0
    })

    with rasterio.open(out_rgb_file, 'w', **pan_meta) as dest_rgb:
        dest_rgb.write(rgb_normalized)
        dest_rgb.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]


def process_raw_delivery_pipeline(input_root, output_dir,
                                  pan_weights=None,
                                  scene_filter=None,
                                  skip_existing=True):
    """
    원본 납품 폴더 구조(중첩)를 재귀 탐색해 일괄 팬샤프닝.

    입력 구조: {input_root}/**/{SCENE}_MUL/ 폴더 안의 *MUL*.tif
               + 형제 폴더    {SCENE}_PAN/ 폴더 안의 *PAN*.tif
      예) mul_data/콩/260710/SM_01_260710_50_MUL/SM_01_260710_50_MUL.tif
          mul_data/밀_보리/260316/30cm/SM_03_260312_30_MUL/SM_03_260312_30_MUL_1.tif
      — 장면 이름은 폴더명({SCENE}_MUL)에서 추출하므로 내부 tif의 `_1` 접미사
        유무와 무관하게 동작.

    출력: {output_dir}/{SCENE}_PANSHARP_8B_1.tif, {SCENE}_CleanStandardRGB_1.tif
      — 후속 crop 단계의 `_` 파싱 규약(마지막 토큰이 일련번호)을 위해 `_1` 접미사 유지.

    :param pan_weights: Pseudo-PAN 가중치. None이면 Legion (현재 납품 위성 기준)
    :param scene_filter: SCENE 이름에 포함돼야 할 문자열 리스트 (None이면 전체)
                         예: ['RL_01', 'SM_01_260710']
    :param skip_existing: True면 8B/RGB 둘 다 이미 존재하는 장면은 건너뜀
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mul_dirs = sorted(glob.glob(
        os.path.join(input_root, "**", "*_MUL"), recursive=True
    ))
    mul_dirs = [d for d in mul_dirs if os.path.isdir(d)]
    if not mul_dirs:
        print(f"⚠️  {input_root} 하위에서 *_MUL 폴더 매칭 없음.")
        return

    n_success, n_skip, n_error = 0, 0, 0
    for mul_dir in mul_dirs:
        scene = os.path.basename(mul_dir)[:-len("_MUL")]

        if scene_filter and not any(key in scene for key in scene_filter):
            continue

        # 폴더 안의 MUL tif (파일명 `_1` 접미사 유무 무관)
        mul_candidates = sorted(glob.glob(os.path.join(mul_dir, "*MUL*.tif")))
        if not mul_candidates:
            print(f"  [스킵] MUL tif 없음: {scene}")
            n_skip += 1
            continue
        ms_path = mul_candidates[0]

        # 형제 _PAN 폴더에서 짝 찾기
        pan_candidates = sorted(glob.glob(os.path.join(
            os.path.dirname(mul_dir), f"{scene}_PAN", "*PAN*.tif"
        )))
        if not pan_candidates:
            print(f"  [스킵] PAN 짝 없음: {scene}")
            n_skip += 1
            continue
        pan_path = pan_candidates[0]

        out_8b_path = os.path.join(output_dir, f"{scene}_PANSHARP_8B_1.tif")
        out_rgb_path = os.path.join(output_dir, f"{scene}_CleanStandardRGB_1.tif")

        if skip_existing and os.path.exists(out_8b_path) and os.path.exists(out_rgb_path):
            print(f"  [스킵] 기존 출력 존재: {scene}")
            n_skip += 1
            continue

        try:
            print(f"\n[처리 중] {scene} (MUL+PAN → 8B/RGB, EPSG:5179)")
            wv3_pansharpen_dual_export_5179(
                ms_path, pan_path, out_8b_path, out_rgb_path, pan_weights=pan_weights
            )
            print(f"  [성공] {os.path.basename(out_8b_path)}, {os.path.basename(out_rgb_path)}")
            n_success += 1
        except Exception as e:
            print(f"  [에러] {scene}: {e}")
            n_error += 1

    print(f"\n[원본 배치 종료] 성공 {n_success} / 스킵 {n_skip} / 오류 {n_error}")


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
    # ============================================================
    # 모드 선택: 'raw'(원본 납품 구조, Legion) / 'flat'(정규화 평면 폴더, 구방식)
    # ============================================================
    MODE = 'raw'

    if MODE == 'raw':
        process_raw_delivery_pipeline(
            input_root="wv_data/mul_data",
            output_dir="wv_data/result_pan",
            pan_weights=PAN_WEIGHTS_LEGION,   # 납품 위성 = WorldView Legion
            scene_filter=None,                # 예: ['RL_01', 'SM_01_260710'] 로 일부만
            skip_existing=True,
        )
    else:
        process_batch_pipeline("wv_data/mul_data", "wv_data/result_pan")