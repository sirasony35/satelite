import rasterio
import rasterio.mask
from rasterio.enums import ColorInterp
import geopandas as gpd
import os
import glob


# ==========================================
# PlanetScope SuperDove 8밴드 순서 (사용자 첨부 기준)
# ==========================================
# band1=CoastalBlue, band2=Blue, band3=GreenI, band4=Green,
# band5=Yellow,      band6=Red,  band7=RedEdge, band8=NIR
# → SENSOR_PRESETS['planetscope_superdove'] = {red=6, nir=8, red_edge=7}
PLANETSCOPE_BAND_ORDER = [
    'CoastalBlue', 'Blue', 'GreenI', 'Green',
    'Yellow', 'Red', 'RedEdge', 'NIR',
]


def crop_field(src, shp_path, output_path):
    """
    SHP 폴리곤으로 raster를 마스킹하여 잘라낸 결과를 원본 dtype·값 그대로 저장.
    팬샤프닝 단계에서 이미 정확한 색감(uint8 RGB)이나 분석용 반사율값(uint16 8밴드)을
    가지고 있으므로 추가 스트레칭·스케일링은 적용하지 않음.
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    nodata_val = src.nodata if src.nodata is not None else 0

    out_image, out_transform = rasterio.mask.mask(
        src, gdf.geometry.values, crop=True, nodata=nodata_val
    )

    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": nodata_val,
        "compress": "lzw",
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def batch_crop_pipeline(tif_dir, shp_dir, output_dir, scene_filter=None):
    """
    [WV 전용] result_pan/의 두 산출물을 모두 필지 단위로 분할.
      - *_CleanStandardRGB*.tif (uint8 3밴드): HPF 보정된 색감 보존
      - *_PANSHARP_8B*.tif       (uint16 8밴드): 식생지수 산출용 원본 반사율 보존

    :param scene_filter: 파일명에 포함돼야 할 문자열 리스트 (None이면 전체)
                         예: ['RL_01_260702', 'SM_01_260710'] 로 일부 장면만 분할
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_files = sorted(glob.glob(os.path.join(tif_dir, "*_CleanStandardRGB*.tif")))
    pan8b_files = sorted(glob.glob(os.path.join(tif_dir, "*_PANSHARP_8B*.tif")))
    shp_files = sorted(glob.glob(os.path.join(shp_dir, "*.shp")))

    if scene_filter:
        rgb_files = [f for f in rgb_files
                     if any(k in os.path.basename(f) for k in scene_filter)]
        pan8b_files = [f for f in pan8b_files
                       if any(k in os.path.basename(f) for k in scene_filter)]

    print(f"[시작] 필지 단위 분할 — RGB {len(rgb_files)}개, 8밴드 {len(pan8b_files)}개, SHP {len(shp_files)}개")

    for tif_path in rgb_files + pan8b_files:
        tif_name = os.path.basename(tif_path).replace(".tif", "")
        parts = tif_name.split("_")

        # 파일명 패턴: SM_<seq>_<date>_<gsd>_<type...>_<n>
        try:
            seq_num = parts[1]
            date_str = parts[2]
            gsd_val = parts[3]
            type_str = "_".join(parts[4:-1])  # CleanStandardRGB 또는 PANSHARP_8B
        except IndexError:
            print(f"  [스킵] 파일명 형식 불일치: {tif_name}")
            continue

        print(f"\n>>> 분할 중: {tif_name}.tif")

        with rasterio.open(tif_path) as src:
            for shp_path in shp_files:
                field_code = os.path.basename(shp_path).replace(".shp", "")
                out_filename = f"{field_code}_{seq_num}_{date_str}_{gsd_val}_{type_str}_Crop.tif"
                output_path = os.path.join(output_dir, out_filename)

                try:
                    crop_field(src, shp_path, output_path)
                    print(f"  [성공] {field_code} → {out_filename}")
                except ValueError:
                    continue  # SHP 영역이 raster 범위 밖일 때
                except Exception as e:
                    print(f"  [에러] {field_code}: {e}")

    print("\n[종료] 모든 필지 분할 작업 완료.")


def crop_polygons_individually(input_tif, shp_path, output_dir,
                                name_field='UID',
                                prefix=None,
                                bands=None,
                                skip_existing=True):
    """
    단일 multi-band raster를 SHP의 각 폴리곤별로 개별 분할.
    한 SHP에 여러 필지 폴리곤이 들어있을 때 사용 (필지당 SHP가 별도가 아닌 경우).

    동작:
      1) SHP의 폴리곤을 raster CRS로 자동 변환
      2) (선택) 특정 밴드만 추출
      3) 각 폴리곤을 개별 mask → 개별 TIF 저장 (원본 dtype·값 그대로 보존)
      4) raster 범위 밖의 폴리곤은 자동 스킵

    :param input_tif: 분할 대상 multi-band raster (예: PlanetScope 8밴드 광역 scene)
    :param shp_path: 필지 경계 SHP (다수 폴리곤 포함)
    :param output_dir: 출력 폴더 — 폴리곤 수만큼 TIF 파일 생성
    :param name_field: SHP 컬럼 중 파일명에 쓸 식별자 (예: 'UID', 'ID', 'PNU').
                       None이거나 컬럼 없으면 인덱스(poly000, poly001 ...) 사용
    :param prefix: 출력 파일명 접두사. None이면 입력 TIF basename 사용
    :param bands: 추출할 밴드 인덱스 리스트 (1-based). None이면 전체 밴드.
                  예: [6, 4, 2] → 3밴드 RGB (PlanetScope SuperDove의 Red=B6, Green=B4, Blue=B2)
                  정확히 3개 밴드 지정 시 자동으로 RGB ColorInterp 설정해
                  QGIS·일반 뷰어에서 트루컬러로 렌더됨.
                  값은 원본 그대로 보존 (스트레칭·스케일링 없음).
    :param skip_existing: True면 이미 존재하는 출력 파일은 건너뜀
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(input_tif))[0]

    gdf = gpd.read_file(shp_path)
    print(f"[폴리곤별 분할 시작] {os.path.basename(input_tif)}")
    print(f"  SHP: {len(gdf)}개 폴리곤, CRS={gdf.crs}")

    n_success, n_skip_outside, n_skip_exist, n_error = 0, 0, 0, 0

    with rasterio.open(input_tif) as src:
        print(f"  TIF: {src.count}밴드 {src.dtypes[0]}, CRS={src.crs}, "
              f"pixel={abs(src.transform[0]):.2f}m")

        # 밴드 선택 검증
        if bands is not None:
            if not all(1 <= b <= src.count for b in bands):
                raise ValueError(f"bands={bands}가 입력 raster 밴드 수({src.count}) 범위를 벗어남")
            print(f"  추출 밴드: {bands} (총 {len(bands)}개)")
            if len(bands) == 3:
                print(f"  → 3밴드 RGB로 ColorInterp 설정 (Red=band{bands[0]}, "
                      f"Green=band{bands[1]}, Blue=band{bands[2]})")

        if gdf.crs != src.crs:
            print(f"  CRS 변환: SHP {gdf.crs} → raster {src.crs}")
            gdf = gdf.to_crs(src.crs)

        nodata_val = src.nodata if src.nodata is not None else 0
        use_field = name_field if (name_field and name_field in gdf.columns) else None
        if name_field and not use_field:
            print(f"  ⚠️  '{name_field}' 컬럼 없음 — 인덱스로 명명. "
                  f"가용 컬럼 일부: {list(gdf.columns)[:8]}")

        for idx, row in gdf.iterrows():
            # 식별자 산출 (파일명 안전 문자만)
            if use_field:
                raw = row[use_field]
                field_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(raw))
            else:
                field_id = f"poly{idx:03d}"

            out_name = f"{prefix}_{field_id}.tif"
            out_path = os.path.join(output_dir, out_name)

            if skip_existing and os.path.exists(out_path):
                n_skip_exist += 1
                continue

            try:
                # bands 지정 시 해당 밴드만 mask·crop (원본 값 그대로 보존)
                out_image, out_transform = rasterio.mask.mask(
                    src, [row.geometry], crop=True, nodata=nodata_val,
                    indexes=bands,  # None이면 전체 밴드, [6,4,2] 같으면 해당 밴드만
                )
            except ValueError:
                # 폴리곤이 raster 범위 밖
                n_skip_outside += 1
                continue
            except Exception as e:
                print(f"  [에러] {field_id}: {e}")
                n_error += 1
                continue

            # out_image: bands 지정 시 (n_bands, H, W), 단일 밴드는 (H, W) → 형태 통일
            if out_image.ndim == 2:
                out_image = out_image[None, :, :]

            n_out_bands = out_image.shape[0]

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "count": n_out_bands,
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": nodata_val,
                "compress": "lzw",
            })
            # 3밴드면 RGB로 photometric 힌트 (QGIS·뷰어가 트루컬러 자동 인식)
            if n_out_bands == 3 and bands is not None:
                out_meta["photometric"] = "RGB"

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(out_image)
                if n_out_bands == 3 and bands is not None:
                    dst.colorinterp = [
                        ColorInterp.red, ColorInterp.green, ColorInterp.blue
                    ]

            n_success += 1
            print(f"  [{n_success}/{len(gdf)}] {field_id} → {out_name} "
                  f"({out_image.shape[2]}x{out_image.shape[1]}, {n_out_bands}밴드)")

    print(f"\n[분할 종료] 성공 {n_success} / 범위 밖 스킵 {n_skip_outside} "
          f"/ 기존 스킵 {n_skip_exist} / 오류 {n_error}")


def run_planetscope_batch(input_folder, shp_dir, output_dir,
                          band_order=PLANETSCOPE_BAND_ORDER,
                          discovery_band='Red',
                          gsd_label='ps',
                          keep_stack=False):
    """
    PlanetScope (SuperDove 8밴드) single-band TIF 묶음을 필지별로 분할.

    동작:
      1) input_folder에서 *_{discovery_band}.tif 기준으로 scene 식별
      2) 각 scene의 8밴드를 multi-band raster로 스택 (드론 stack 로직 재사용)
      3) shp_dir의 모든 SHP로 분할 → output_dir에 8-band uint16 보존하며 저장

    입력 파일명 규약: {base}_{BAND}.tif
      예) 20250728_CoastalBlue.tif, 20250728_Blue.tif, ..., 20250728_NIR.tif
      → base = '20250728', BAND in band_order

    출력 파일명: {field}_{base}_{gsd_label}_PS_8B_Crop.tif
      예) SM01_20250728_ps_PS_8B_Crop.tif

    detect_gaps 호출 시: SENSOR_PRESETS['planetscope_superdove']
      = {band_red=6, band_nir=8, band_red_edge=7}

    :param band_order: 스택 시 밴드 순서. 출력 multi-band raster의 1~N 인덱스
    :param discovery_band: scene 식별에 사용할 밴드 키워드 (기본 'Red')
    :param gsd_label: 출력 파일명 라벨 ('ps', '3m', 'planetscope' 등)
    :param keep_stack: True면 스택 raster를 output_dir/_stack/ 에 보관 (디버깅용)
    """
    # 드론과 동일한 스택 로직 재사용 — wv3_detecting_gaps에 이미 구현됨
    from wv3_detecting_gaps import stack_drone_bands, DRONE_STACK_EXT

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenes = sorted(glob.glob(os.path.join(input_folder, f"*_{discovery_band}.tif")))
    if not scenes:
        print(f"⚠️  {input_folder}/*_{discovery_band}.tif 매칭 없음. "
              f"파일명이 '{{base}}_{discovery_band}.tif' 패턴인지 확인하세요.")
        return

    shp_files = sorted(glob.glob(os.path.join(shp_dir, "*.shp")))
    if not shp_files:
        print(f"⚠️  {shp_dir}/*.shp 매칭 없음.")
        return

    print(f"[PlanetScope 배치 시작] {len(scenes)}개 scene × {len(shp_files)}개 필지")
    print(f"           band_order={band_order}")
    print(f"           스택 형식={DRONE_STACK_EXT}, gsd_label='{gsd_label}'")

    stack_dir = os.path.join(output_dir, '_stack')
    if not os.path.exists(stack_dir):
        os.makedirs(stack_dir)

    n_success, n_skip, n_error = 0, 0, 0

    for i, red_file in enumerate(scenes, 1):
        base = os.path.basename(red_file).replace(f'_{discovery_band}.tif', '')
        print(f"\n----- [{i}/{len(scenes)}] scene: {base} -----")

        # 1) 8밴드 스택
        stack_path_req = os.path.join(stack_dir, f"{base}_STACK{DRONE_STACK_EXT}")
        try:
            stack_path = stack_drone_bands(input_folder, base, stack_path_req, band_order)
        except FileNotFoundError as e:
            print(f"❌ 스택 실패 — 누락 밴드: {e}")
            n_error += 1
            continue
        print(f"   - 스택 생성: {os.path.basename(stack_path)}")

        # 2) 각 SHP로 분할
        with rasterio.open(stack_path) as src:
            print(f"   - 스택 정보: {src.count}밴드, {src.dtypes[0]}, "
                  f"{src.width}x{src.height}, pixel={abs(src.transform[0]):.3f}m")
            for shp_path in shp_files:
                field = os.path.basename(shp_path).replace('.shp', '')
                out_name = f"{field}_{base}_{gsd_label}_PS_8B_Crop.tif"
                out_path = os.path.join(output_dir, out_name)
                try:
                    crop_field(src, shp_path, out_path)
                    print(f"  [성공] {field} → {out_name}")
                    n_success += 1
                except ValueError:
                    # SHP 영역이 raster 범위 밖
                    n_skip += 1
                    continue
                except Exception as e:
                    print(f"  [에러] {field}: {e}")
                    n_error += 1

        # 3) 임시 스택 정리
        if not keep_stack and os.path.exists(stack_path):
            try:
                os.remove(stack_path)
            except OSError:
                pass

    # _stack 폴더가 비었으면 정리
    if not keep_stack:
        try:
            if os.path.isdir(stack_dir) and not os.listdir(stack_dir):
                os.rmdir(stack_dir)
        except OSError:
            pass

    print(f"\n[PlanetScope 배치 종료] 성공 {n_success} / 스킵 {n_skip} / 오류 {n_error}")


# ==========================================
# 실행부 (MODE 스위치)
# ==========================================
if __name__ == "__main__":

    # ============================================================
    # 모드 선택: 'wv3' / 'planetscope' / 'pcc' / 'pcc_rgb'
    # ============================================================
    MODE = 'pcc_rgb'

    if MODE == 'wv3':
        # ── WV3: result_pan/ → crop_result/ ──
        batch_crop_pipeline(
            tif_dir="wv_data/result_pan",
            shp_dir="wv_data/Shapefile",
            output_dir="wv_data/crop_result",
        )

    elif MODE == 'planetscope':
        # ── PlanetScope: 8개 single-band TIF → 필지별 8-band Crop ──
        # 입력 파일명: {base}_CoastalBlue.tif, _Blue.tif, _GreenI.tif, _Green.tif,
        #              _Yellow.tif, _Red.tif, _RedEdge.tif, _NIR.tif
        # 출력 파일명: {field}_{base}_{gsd_label}_PS_8B_Crop.tif
        # 이후 wv3_detecting_gaps.py에서 SENSOR_PRESETS['planetscope_superdove']로 분석
        run_planetscope_batch(
            input_folder="wv_data/planetscope_data",  #경로변경
            shp_dir="wv_data/Shapefile", # 경로변경
            output_dir="wv_data/planetscope_crop_result",
            band_order=PLANETSCOPE_BAND_ORDER,
            discovery_band='Red',     # scene 식별 기준
            gsd_label='ps',           # 출력 파일명에 들어갈 라벨
            keep_stack=False,         # True면 스택 raster 보관 (디버깅용)
        )

    elif MODE == 'pcc':
        # ── PCC: 단일 multi-band TIF + 단일 SHP(여러 폴리곤) → 폴리곤별 개별 TIF ──
        # 입력: 1개 multi-band TIF + 1개 SHP(N개 필지 포함)
        # 출력: pcc/crop_result/ 에 N개 TIF (필지 식별자별로 명명)
        # 좌표계 다르면 자동 변환. raster 범위 밖 폴리곤은 자동 스킵.
        crop_polygons_individually(
            input_tif="pcc/data/260615_Yeoncheon_Daedong_No.1_SuperX.tif",
            shp_path="pcc/Shapefile/연천필지/연천 필지.shp",
            output_dir="pcc/crop_result",
            name_field='UID',         # 필지 식별 컬럼 (8자리 UID). 다른 후보: 'ID', 'PNU'
            prefix='Yeoncheon_260615', # 출력 파일명 접두사 (None이면 입력 TIF basename)
            skip_existing=True,
        )

    elif MODE == 'pcc_rgb':
        # ── PCC RGB: 8밴드 PlanetScope에서 RGB(Red=B6, Green=B4, Blue=B2)만 추출 + 필지 분할 ──
        # 원본 uint16 값 그대로 보존 (스트레칭·스케일링 없음 → 색상 안 틀어짐)
        # 3밴드 RGB로 ColorInterp 자동 설정 → QGIS에서 트루컬러 자동 렌더
        crop_polygons_individually(
            input_tif="pcc/data/260615_Yeoncheon_Daedong_No.1_SuperX.tif",
            shp_path="pcc/Shapefile/연천필지/연천 필지.shp",
            output_dir="pcc/crop_result_rgb",
            name_field='UID',
            prefix='Yeoncheon_260615_RGB',
            bands=[6, 4, 2],          # Red=B6, Green=B4, Blue=B2 (PlanetScope SuperDove)
            skip_existing=True,
        )

    else:
        raise ValueError(
            f"알 수 없는 MODE: {MODE}. 'wv3'/'planetscope'/'pcc'/'pcc_rgb' 중 선택."
        )
