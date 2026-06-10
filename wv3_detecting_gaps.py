import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
import warnings

# Windows cp949 콘솔에서도 한글·특수문자(—, →, ⚠ 등) 출력 가능하도록 stdout UTF-8 재설정
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# matplotlib는 PNG 시각화 전용 — 미설치 시 PNG 단계만 스킵하고 SHP/TIF는 정상 출력
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from rasterio.plot import show as rio_show
    MATPLOTLIB_AVAILABLE = True

    # 한글 폰트 직접 등록 (matplotlib 캐시 stale이어도 시스템 파일 경로로 강제 추가)
    _font_paths = [
        r'C:\Windows\Fonts\malgun.ttf',                      # Windows 맑은 고딕
        r'C:\Windows\Fonts\malgunbd.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',        # macOS
        '/Library/Fonts/AppleGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',   # Linux nanum
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    KOREAN_FONT = None  # FontProperties 객체 — 각 text() 호출에 fontproperties로 전달
    for _p in _font_paths:
        if os.path.exists(_p):
            try:
                fm.fontManager.addfont(_p)
                KOREAN_FONT = fm.FontProperties(fname=_p)
                plt.rcParams['font.family'] = KOREAN_FONT.get_name()
                break
            except Exception:
                continue
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    KOREAN_FONT = None

warnings.filterwarnings('ignore')


# ==========================================
# 센서별 밴드 인덱스 프리셋 (1-based)
# ==========================================
# 사용 예: detect_missing_plants(..., **SENSOR_PRESETS['dji_mavic3m'])
# 새 센서 추가 시 (Red, NIR, RedEdge) 인덱스만 지정하면 됨.
SENSOR_PRESETS = {
    # WorldView-3 위성 8밴드 (Coastal/Blue/Green/Yellow/Red/RedEdge/NIR1/NIR2)
    'wv3':                dict(band_red=5, band_nir=7, band_red_edge=6),

    # MicaSense RedEdge / Altum (Blue/Green/Red/NIR/RedEdge) — 5밴드 멀티스펙트럴
    'micasense_rededge':  dict(band_red=3, band_nir=4, band_red_edge=5),
    'micasense_altum':    dict(band_red=3, band_nir=4, band_red_edge=5),

    # DJI Phantom 4 Multispectral (Blue/Green/Red/RedEdge/NIR) — 5밴드
    'dji_phantom4_ms':    dict(band_red=3, band_nir=5, band_red_edge=4),

    # DJI Mavic 3M (Green/Red/RedEdge/NIR) — 4밴드, Blue 없음
    'dji_mavic3m':        dict(band_red=2, band_nir=4, band_red_edge=3),

    # Sentera 6X (Blue/Green/Red/RedEdge/NIR/...) — 5~6밴드
    'sentera_6x':         dict(band_red=3, band_nir=5, band_red_edge=4),

    # Parrot Sequoia (Green/Red/RedEdge/NIR) — 4밴드
    'parrot_sequoia':     dict(band_red=2, band_nir=4, band_red_edge=3),
}


# ==========================================
# Otsu 분포 품질 (이중모드 신뢰도)
# ==========================================
def otsu_bimodality_quality(values, threshold):
    """
    Otsu 임계값에서의 between-class variance 비율을 산출.
    값 범위 [0, 1] — 높을수록 분포가 명확히 이중모드.

      Q >= 0.5 : 매우 강한 이중모드 (작물·토양 명확 분리)
      0.3~0.5  : 적정 분리 (분류 신뢰 가능)
      < 0.3    : 단봉분포 또는 약한 분리 (임계가 인위적 — 휴경/수확후 또는 균질 영상)

    절대 NDVI 임계와 달리 작물 식생의 절대 강도에 무관하므로 새만금 간척지처럼
    식생 NDVI가 본래 낮은 환경에서도 정상 동작.
    """
    low = values[values <= threshold]
    high = values[values > threshold]
    if len(low) == 0 or len(high) == 0:
        return 0.0
    n = len(values)
    w0, w1 = len(low) / n, len(high) / n
    mu0, mu1 = low.mean(), high.mean()
    mu = values.mean()
    sigma_b2 = w0 * (mu0 - mu) ** 2 + w1 * (mu1 - mu) ** 2
    sigma_total = values.var()
    return float(sigma_b2 / sigma_total) if sigma_total > 1e-12 else 0.0


# ==========================================
# 식생지수 산출
# ==========================================
def compute_vegetation_index(src, method='msavi2',
                              band_red=5, band_nir=7, band_red_edge=6):
    """
    다중분광 raster에서 식생지수 산출. 밴드 인덱스를 노출해 WV3/드론 무관 동작.

    method 선택:
      - 'msavi2' (기본): 토양 보정선 자체 산출.
                   염류토(간척지) 또는 사질토 배경에서 NDVI 부풀림 보정.
                   파종~영양생장기처럼 토양 노출 많은 시기에 특히 강함.
      - 'ndvi'   : (NIR - Red) / (NIR + Red).
                   표준·scale-invariant. 폐쇄 캐노피기에는 MSAVI2와 거의 동등.
      - 'ndre'   : (NIR - RedEdge) / (NIR + RedEdge).
                   Red Edge 사용 — 엽록소 민감, 포화 없음.
                   폐쇄 캐노피기 활력도·스트레스 단계 구분에 우수.

    :param band_red:      Red 밴드 인덱스 (1-based). WV3=5, MicaSense=3, DJI Mavic3M=2 등
    :param band_nir:      NIR 밴드 인덱스 (1-based)
    :param band_red_edge: RedEdge 밴드 인덱스 (1-based) — NDRE에서만 사용
    :return: (index_array_with_NaN_outside_field, field_mask)
    """
    nodata_val = src.nodata if src.nodata is not None else 0

    if src.count < max(band_red, band_nir):
        raise ValueError(
            f"입력 raster 밴드 수 {src.count}개가 필요한 인덱스(red={band_red}, "
            f"nir={band_nir})에 부족합니다. SENSOR_PRESETS 확인 또는 직접 지정."
        )

    red = src.read(band_red).astype(np.float32)
    nir = src.read(band_nir).astype(np.float32)
    field_mask = (red != nodata_val) & (nir != nodata_val)

    if method == 'ndvi':
        denom = nir + red
        index = np.where(denom > 0, (nir - red) / denom, 0.0)
    elif method == 'msavi2':
        radicand = (2 * nir + 1) ** 2 - 8 * (nir - red)
        radicand = np.where(radicand < 0, 0, radicand)
        index = (2 * nir + 1 - np.sqrt(radicand)) / 2
    elif method == 'ndre':
        if src.count < band_red_edge:
            raise ValueError(
                f"NDRE 사용 시 RedEdge 밴드 인덱스 {band_red_edge}가 필요한데 "
                f"입력 밴드 수는 {src.count}개입니다."
            )
        red_edge = src.read(band_red_edge).astype(np.float32)
        field_mask = field_mask & (red_edge != nodata_val)
        denom = nir + red_edge
        index = np.where(denom > 0, (nir - red_edge) / denom, 0.0)
    else:
        raise ValueError(
            f"지원하지 않는 method: '{method}'. 'ndvi'/'msavi2'/'ndre' 중 선택"
        )

    index = np.where(field_mask, index, np.nan)
    return index, field_mask


# ==========================================
# PNG 시각화 (RGB 배경 + 결주 폴리곤 오버레이)
# ==========================================
def render_gap_png(rgb_path, gdf, title_text, info_lines, output_path):
    """
    배경 RGB 위에 결주 폴리곤을 오버레이한 PNG를 생성.

    레이아웃:
      - 상단 제목 (필지/날짜/해상도)
      - 중앙: RGB 배경 + 결주 폴리곤(주황 반투명 + 빨간 외곽). 폴리곤 면적 라벨은 표시 안 함.
      - 하단(이미지 외부): 통계 박스 (이미지와 겹치지 않도록 figure 좌표계 사용)

    :param rgb_path: 배경 RGB raster 경로
    :param gdf: 결주 폴리곤 GeoDataFrame (비어있어도 OK)
    :param title_text: 상단 제목
    :param info_lines: 하단 통계 박스에 표시할 텍스트 라인 리스트
    :param output_path: 저장할 PNG 경로
    """
    if not MATPLOTLIB_AVAILABLE:
        print("   [skip] matplotlib 미설치 — PNG 생략")
        return False

    if not os.path.exists(rgb_path):
        print(f"   [skip] RGB 파일 없음 — PNG 생략: {os.path.basename(rgb_path)}")
        return False

    # 이미지(axes) 위·아래에 명시적 여백 → 하단 정보박스가 이미지와 겹치지 않음
    fig = plt.figure(figsize=(12, 11))
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.80])  # [left, bottom, width, height]

    with rasterio.open(rgb_path) as rgb_src:
        rio_show(rgb_src, ax=ax)

    # 한글 fontproperties — 모든 텍스트 요소에 명시적으로 전달해 폰트 누락 방지
    fp_kw = {'fontproperties': KOREAN_FONT} if KOREAN_FONT else {}

    if (gdf is not None) and (not gdf.empty):
        # 결주 폴리곤 = 주황 반투명 + 빨간 외곽 (면적 라벨은 표시 안 함)
        gdf.plot(ax=ax, facecolor='orange', edgecolor='red', linewidth=0.8, alpha=0.55)

    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15, **fp_kw)
    ax.set_axis_off()

    # 통계 박스 — figure 좌표로 axes 아래 별도 영역에 배치 (이미지와 절대 겹치지 않음)
    fig.text(
        0.5, 0.06, '\n'.join(info_lines),
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                  alpha=0.92, edgecolor='gray'),
        **fp_kw,
    )

    # tight_layout / bbox_inches='tight' 금지 — 위에서 잡은 add_axes 위치를 망가뜨림
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


# ==========================================
# 결주 탐지 본체
# ==========================================
def detect_missing_plants(
    input_filepath,
    output_dir,
    *,
    # === 식생지수 설정 (센서 종류) ===
    index_method='msavi2',     # 'msavi2'(기본) / 'ndvi' / 'ndre'
    band_red=5,                # Red 밴드 인덱스 (1-based). WV3=5, drone은 SENSOR_PRESETS 참고
    band_nir=7,                # NIR 밴드 인덱스
    band_red_edge=6,           # RedEdge 밴드 인덱스 (NDRE 사용 시만)

    # === 결주 검출 파라미터 (작물 종류·해상도) ===
    min_gap_area_sqm=0.5,      # 결주 최소 면적(㎡)
    closing_radius=None,       # None = 행간·픽셀 크기로 자동 계산
    row_spacing_m=0.65,        # 작물 행간(m). 논콩 0.60~0.70

    # === 출력 네이밍 (None이면 WV3 SM_XX_YYMMDD_GSD 패턴 파싱) ===
    field_code=None,
    date_str=None,
    gsd_label=None,

    # === PNG 배경 RGB (None이면 WV3 PANSHARP_8B_Crop→CleanStandardRGB_Crop 자동 변환) ===
    rgb_path=None,
):
    """
    다중분광 raster에서 식생지수를 계산하고 결주 구역을 폴리곤으로 추출.
    WV3 위성·MicaSense·DJI 등 모든 다중분광 센서에서 동일 알고리즘 적용 가능.

    산출물(output_dir):
      - {field}_{date}_{gsd}_GAPS.shp       — 결주 폴리곤 (벡터 처방지도)
      - {field}_{date}_{gsd}_{INDEX}.tif    — 식생지수 raster (QGIS 검증용)
      - {field}_{date}_{gsd}_GAPS.png       — RGB 위 결주 오버레이 (rgb_path 있을 때)

    closing_radius=None 시 자동 산출: round(row_spacing_m / pixel_size_m / 2),
    범위 [1, 30]. 행간 30cm 미만 작물은 row_spacing_m을 직접 줄여 지정.

    :return: 추출된 결주 SHP 경로 (또는 유효 픽셀 없을 시 None)
    """
    # === 파일명에서 메타 추출 (override 없을 때) ===
    filename = os.path.basename(input_filepath)
    name_parts = filename.replace('.tif', '').split('_')
    if field_code is None or date_str is None or gsd_label is None:
        if len(name_parts) >= 4:
            field_code = field_code or name_parts[0]
            date_str = date_str or name_parts[2]
            gsd_label = gsd_label or name_parts[3]
        else:
            raise ValueError(
                f"파일명에서 field/date/gsd 추출 실패: {filename}. "
                f"field_code, date_str, gsd_label 파라미터로 명시 지정 필요."
            )

    index_upper = index_method.upper()
    out_shp_name = f"{field_code}_{date_str}_{gsd_label}_GAPS.shp"
    out_idx_name = f"{field_code}_{date_str}_{gsd_label}_{index_upper}.tif"
    out_png_name = f"{field_code}_{date_str}_{gsd_label}_GAPS.png"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[알림] 결과 저장 폴더 생성: {output_dir}")

    shp_filepath = os.path.join(output_dir, out_shp_name)
    idx_filepath = os.path.join(output_dir, out_idx_name)
    png_filepath = os.path.join(output_dir, out_png_name)

    print(f"\n[{field_code} / {date_str} / {gsd_label} / {index_upper}] 결주 분석 시작")

    with rasterio.open(input_filepath) as src:
        crs = src.crs
        transform = src.transform

        pixel_area_sqm = abs(transform[0] * transform[4])
        pixel_size_m = pixel_area_sqm ** 0.5
        min_pixels = max(int(min_gap_area_sqm / pixel_area_sqm), 1)

        # closing_radius 자동 산출: 행간을 두 번의 dilate-erode로 메우는 보수적 반경
        # 식: round(행간 / 픽셀크기 / 2), 범위 [1, 10]
        # - WV3 30cm + 행간 65cm: round(1.08) = 1px
        # - 드론  5cm + 행간 65cm: round(6.5)  = 7px
        # - 드론 10cm + 행간 65cm: round(3.25) = 3px
        # 행간이 캐노피로 완전히 덮인 영상은 더 작게, 행이 또렷이 보이면 더 크게 명시 권장
        if closing_radius is None:
            closing_radius_used = max(1, min(int(round(row_spacing_m / pixel_size_m / 2)), 10))
            print(f"   - closing_radius 자동: {closing_radius_used}px "
                  f"(행간 {row_spacing_m}m / 픽셀 {pixel_size_m:.3f}m / 2)")
        else:
            closing_radius_used = closing_radius

        index, field_mask = compute_vegetation_index(
            src, method=index_method,
            band_red=band_red, band_nir=band_nir, band_red_edge=band_red_edge,
        )

        if not field_mask.any():
            print("⚠️  필지 내부 유효 픽셀이 없습니다.")
            return None

        valid_idx = index[field_mask]
        threshold = threshold_otsu(valid_idx)
        quality = otsu_bimodality_quality(valid_idx, threshold)

        # 분류 신뢰도 라벨 (절대 NDVI/MSAVI2 값과 무관 — 새만금 간척지 식생 적합)
        if quality >= 0.5:
            quality_label = "강한 이중모드 — 매우 신뢰"
        elif quality >= 0.3:
            quality_label = "적정 이중모드 — 신뢰 가능"
        else:
            quality_label = "단봉/약한 분리 — 의심 (휴경/수확후 가능성)"

        field_area = field_mask.sum() * pixel_area_sqm
        print(f"   - 필지 면적: {field_area:,.1f}㎡ ({int(field_mask.sum()):,}px)")
        print(f"   - {index_upper} 분포: min={valid_idx.min():.3f}, "
              f"median={np.median(valid_idx):.3f}, max={valid_idx.max():.3f}")
        print(f"   - Otsu 임계값: {threshold:.4f}")
        print(f"   - 분포 품질 Q: {quality:.3f}  →  {quality_label}")

        # 식생/결주 분리 + 행간 closing
        plant_mask = (index > threshold) & field_mask
        if closing_radius_used > 0:
            plant_mask = closing(plant_mask, disk(closing_radius_used)) & field_mask
        gap_mask = (~plant_mask) & field_mask

        # 최소 면적 필터
        filtered_gaps = remove_small_objects(gap_mask, min_size=min_pixels)

        gap_pixels = int(filtered_gaps.sum())
        gap_area = gap_pixels * pixel_area_sqm
        gap_ratio = (gap_area / field_area * 100) if field_area > 0 else 0.0
        print(f"   - 결주 면적: {gap_area:,.1f}㎡ ({gap_pixels:,}px), 결주율 {gap_ratio:.2f}%")

        # 식생지수 raster 저장
        idx_meta = src.meta.copy()
        idx_meta.update(count=1, dtype='float32', nodata=np.nan, compress='lzw')
        with rasterio.open(idx_filepath, 'w', **idx_meta) as dst:
            dst.write(index.astype('float32'), 1)

        # 결주 폴리곤 벡터화
        filtered_gaps_uint = filtered_gaps.astype(np.uint8)
        geom_results = (
            {'properties': {'class': 'Gap'}, 'geometry': s}
            for s, _ in shapes(filtered_gaps_uint,
                               mask=(filtered_gaps_uint == 1),
                               transform=transform)
        )
        geometries = list(geom_results)

        if geometries:
            gdf = gpd.GeoDataFrame.from_features(geometries, crs=crs)
            gdf['area_sqm'] = gdf.geometry.area.round(2)
            # SHP 속성에 분포 품질 기록 — QGIS에서 신뢰도 낮은 결과 필터링용
            gdf['otsu_qual'] = round(quality, 3)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.1, preserve_topology=True)
            gdf.to_file(shp_filepath, encoding='utf-8')
            print(f"✅ 결주 SHP:   {out_shp_name} ({len(gdf)}개 폴리곤)")
        else:
            gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
            print("✅ 탐지된 결주 구역 없음 — SHP 생성 생략")

        print(f"   {index_upper} raster: {out_idx_name}")

    # PNG 시각화 — rgb_path 지정 없으면 WV3 패턴(PANSHARP_8B_Crop→CleanStandardRGB_Crop) 자동 변환
    if rgb_path is None:
        rgb_path = input_filepath.replace('PANSHARP_8B_Crop', 'CleanStandardRGB_Crop')
    title_text = f"{field_code}   /   {date_str}   /   {gsd_label}"
    quality_mark = "[OK]" if quality >= 0.3 else "[!]"
    info_lines = [
        f"지수: {index_upper}  (Otsu {threshold:.3f})",
        f"분포품질: {quality:.3f}  {quality_mark} {quality_label}",
        f"필지 면적: {field_area:,.1f} ㎡",
        f"결주 면적: {gap_area:,.1f} ㎡",
        f"결주율: {gap_ratio:.2f} %",
        f"폴리곤: {len(gdf)} 개",
    ]
    ok = render_gap_png(rgb_path, gdf, title_text, info_lines, png_filepath)
    if ok:
        print(f"   PNG     : {out_png_name}")

    return shp_filepath


# ==========================================
# 드론 single-band TIF → 다중밴드 raster 스택
# ==========================================
DRONE_BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'REDEDGE', 'NIR']
# 위 순서로 스택 시 밴드 인덱스 매핑 (1-based):
#   1=Blue, 2=Green, 3=Red, 4=RedEdge, 5=NIR
# → SENSOR_PRESETS['micasense_rededge'] 와 동일 매핑이라 그대로 재사용 가능

# osgeo 가용성 — VRT(가상 raster, ~1KB) vs multi-band GeoTIFF(실제 데이터 복제) 결정
try:
    from osgeo import gdal as _gdal_check
    _gdal_check.UseExceptions()
    HAS_OSGEO = True
    DRONE_STACK_EXT = '.vrt'  # 가상 raster, 디스크 절약
except ImportError:
    HAS_OSGEO = False
    DRONE_STACK_EXT = '.tif'  # multi-band GeoTIFF (osgeo 없을 때)


def stack_drone_bands(folder, base_name, output_path,
                       band_order=DRONE_BAND_ORDER):
    """
    드론의 single-band TIF 5개를 multi-band raster로 묶음.

    파일명 패턴: {base_name}_{BAND}.tif (예: SM01_01_250728_RED.tif)

    output_path 확장자가 .vrt면 GDAL VRT(가상 raster, ~1KB) 생성.
    그 외(예: .tif)는 multi-band GeoTIFF로 실제 데이터 복제.

    osgeo 미설치 시 .vrt 요청도 GeoTIFF 폴백.
    """
    files = [os.path.join(folder, f"{base_name}_{b}.tif") for b in band_order]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"누락된 밴드: {[os.path.basename(m) for m in missing]}"
        )

    if output_path.lower().endswith('.vrt'):
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            opts = gdal.BuildVRTOptions(separate=True)
            ds = gdal.BuildVRT(output_path, files, options=opts)
            # 명시적 close 필수 — 안 하면 디스크 플러시 안 됨
            ds.FlushCache()
            ds = None
            return output_path
        except ImportError:
            # osgeo 없으면 .vrt 확장자라도 GeoTIFF로 폴백
            output_path = output_path[:-4] + '.tif'

    # GeoTIFF multi-band 스택
    with rasterio.open(files[0]) as src0:
        meta = src0.meta.copy()
        meta.update(count=len(band_order), compress='lzw')

    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, f in enumerate(files, 1):
            with rasterio.open(f) as s:
                dst.write(s.read(1), i)

    return output_path


def run_drone_batch(input_folder, result_folder,
                     band_order=DRONE_BAND_ORDER,
                     sensor_params=None,
                     index_method='msavi2',
                     min_gap_area_sqm=0.5,
                     closing_radius=None,
                     row_spacing_m=0.65,
                     gsd_label='drone',
                     keep_stack=False):
    """
    드론 single-band TIF 묶음을 일괄 처리.

    동작:
      1) 폴더 내 *_RED.tif를 기준으로 필지 식별
      2) 각 필지의 5밴드를 VRT로 묶음
      3) 같은 base_name의 *_RGB.tif가 있으면 PNG 배경으로 사용
      4) detect_missing_plants 호출

    :param band_order: 스택 시 밴드 순서. 기본 [BLUE,GREEN,RED,REDEDGE,NIR]
    :param sensor_params: dict(band_red, band_nir, band_red_edge).
                          None이면 band_order 기준 자동 계산
    :param gsd_label: 출력 파일명에 들어갈 해상도 라벨 (예: 'drone', '4cm')
    :param keep_stack: True면 VRT 보관, False면 처리 후 삭제
    """
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # band_order 기반 자동 sensor_params (각 밴드의 출력 인덱스)
    if sensor_params is None:
        try:
            sensor_params = dict(
                band_red=band_order.index('RED') + 1,
                band_nir=band_order.index('NIR') + 1,
                band_red_edge=band_order.index('REDEDGE') + 1,
            )
        except ValueError as e:
            raise ValueError(
                f"band_order에 'RED'/'NIR'/'REDEDGE' 모두 포함 필요: {band_order}"
            ) from e

    red_files = sorted(glob.glob(os.path.join(input_folder, "*_RED.tif")))
    if not red_files:
        print(f"⚠️  {input_folder}/*_RED.tif 매칭 없음.")
        return

    print(f"[드론 배치 시작] {len(red_files)}개 필지")
    print(f"           band_order={band_order} → sensor={sensor_params}")
    print(f"           index={index_method}, row_spacing={row_spacing_m}m, "
          f"min_gap={min_gap_area_sqm}㎡, "
          f"closing={closing_radius if closing_radius is not None else '자동'}")

    stack_dir = os.path.join(result_folder, '_stack')
    if not os.path.exists(stack_dir):
        os.makedirs(stack_dir)

    n_success, n_skip, n_error = 0, 0, 0

    for i, red_file in enumerate(red_files, 1):
        base = os.path.basename(red_file).replace('_RED.tif', '')
        print(f"\n----- [{i}/{len(red_files)}] {base} -----")

        # 1) 5밴드 → VRT(osgeo 있음) 또는 multi-band GeoTIFF
        stack_path_req = os.path.join(stack_dir, f"{base}_STACK{DRONE_STACK_EXT}")
        try:
            # 반환값이 실제 생성된 경로 (osgeo 없으면 .tif 폴백)
            stack_path = stack_drone_bands(input_folder, base, stack_path_req, band_order)
        except FileNotFoundError as e:
            print(f"❌ 스택 실패: {e}")
            n_error += 1
            continue
        print(f"   - 스택 생성: {os.path.basename(stack_path)} "
              f"({'VRT' if stack_path.endswith('.vrt') else 'GeoTIFF'})")

        # 2) RGB 배경 자동 인식
        rgb_candidate = os.path.join(input_folder, f"{base}_RGB.tif")
        rgb = rgb_candidate if os.path.exists(rgb_candidate) else None

        # 3) base_name에서 field/date 추출 (예: SM01_01_250728)
        parts = base.split('_')
        if len(parts) < 3:
            print(f"⚠️  파일명 형식 불일치: {base}")
            n_skip += 1
            continue
        field_code, date_str = parts[0], parts[2]

        # 4) 결주 분석
        try:
            result = detect_missing_plants(
                input_filepath=stack_path,
                output_dir=result_folder,
                index_method=index_method,
                min_gap_area_sqm=min_gap_area_sqm,
                closing_radius=closing_radius,
                row_spacing_m=row_spacing_m,
                field_code=field_code,
                date_str=date_str,
                gsd_label=gsd_label,
                rgb_path=rgb,
                **sensor_params,
            )
            if result is None:
                n_skip += 1
            else:
                n_success += 1
        except Exception as e:
            n_error += 1
            print(f"❌ 분석 실패: {e}")

        # 5) 임시 스택 정리
        if not keep_stack and os.path.exists(stack_path):
            try:
                # VRT은 동시에 .ovr 등 생성 안 됨, 안전하게 단일 파일 삭제
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

    print(f"\n[드론 배치 종료] 성공 {n_success} / 스킵 {n_skip} / 오류 {n_error}")


# ==========================================
# 배치 처리 (재사용 함수)
# ==========================================
def run_batch(input_folder, result_folder, file_pattern, sensor_params,
              index_method='msavi2', min_gap_area_sqm=0.5, closing_radius=None,
              row_spacing_m=0.65, rgb_resolver=None):
    """
    폴더 내 모든 입력 파일을 일괄 처리.

    :param sensor_params: SENSOR_PRESETS[key] 또는 dict(band_red=..., band_nir=..., band_red_edge=...)
    :param rgb_resolver: callable(input_path) -> rgb_path. None이면 자동 (WV3 패턴) 또는 None
    """
    input_files = sorted(glob.glob(os.path.join(input_folder, file_pattern)))
    if not input_files:
        print(f"⚠️  {input_folder}/{file_pattern} 매칭 없음.")
        return

    print(f"[배치 시작] {len(input_files)}개 파일")
    print(f"           index={index_method}, sensor={sensor_params}, "
          f"row_spacing={row_spacing_m}m, min_gap={min_gap_area_sqm}㎡, "
          f"closing={closing_radius if closing_radius is not None else '자동'}")
    n_success, n_skip, n_error = 0, 0, 0

    for i, input_file in enumerate(input_files, 1):
        print(f"\n----- [{i}/{len(input_files)}] {os.path.basename(input_file)} -----")
        try:
            rgb = rgb_resolver(input_file) if rgb_resolver else None
            result = detect_missing_plants(
                input_filepath=input_file,
                output_dir=result_folder,
                index_method=index_method,
                min_gap_area_sqm=min_gap_area_sqm,
                closing_radius=closing_radius,
                row_spacing_m=row_spacing_m,
                rgb_path=rgb,
                **sensor_params,
            )
            if result is None:
                n_skip += 1
            else:
                n_success += 1
        except Exception as e:
            n_error += 1
            print(f"❌ 오류: {e}")

    print(f"\n[배치 종료] 성공 {n_success} / 스킵 {n_skip} / 오류 {n_error}")


# ==========================================
# 실행부 (모드 스위치)
# ==========================================
if __name__ == "__main__":

    # ============================================================
    # 모드 선택: 'wv3' (위성, 새만금) 또는 'drone' (드론, 육지)
    # ============================================================
    MODE = 'wv3'

    if MODE == 'wv3':
        # ── WV3 위성 — 새만금 간척지 + 논콩 ──
        run_batch(
            input_folder="wv_data/crop_result",
            result_folder="wv_data/result_gaps",
            file_pattern="*_PANSHARP_8B_Crop.tif",
            sensor_params=SENSOR_PRESETS['wv3'],
            index_method='msavi2',       # 간척지 염류토는 MSAVI2 권장
            min_gap_area_sqm=0.5,
            closing_radius=2,            # 30cm GSD 명시 (자동도 약 2가 나옴)
            row_spacing_m=0.65,          # 논콩
        )

    elif MODE == 'drone':
        # ── 드론 다중분광 — single-band TIF 묶음 처리 ──
        # 입력 폴더에 {base}_BLUE.tif / _GREEN.tif / _RED.tif / _REDEDGE.tif / _NIR.tif
        # 가 한 세트로 존재해야 함. {base}_RGB.tif 가 있으면 PNG 배경으로 자동 사용.
        #
        # row_spacing_m: 작물 행간 (논콩 0.65, 옥수수 0.75, 양파 0.18 등)
        # closing_radius=None: 자동 산출 (4cm 해상도 + 65cm 행간 → 약 8px)
        # gsd_label: 출력 파일명에 들어갈 라벨 ('drone', '4cm' 등)
        run_drone_batch(
            input_folder="wv_data/drone_data",
            result_folder="wv_data/drone_result_gaps",
            band_order=['BLUE', 'GREEN', 'RED', 'REDEDGE', 'NIR'],  # 스택 순서
            index_method='msavi2',   # 'ndvi' 비교도 권장
            min_gap_area_sqm=0.5,
            closing_radius=None,     # 자동 (4cm + 65cm → 약 8px)
            row_spacing_m=0.65,      # 논콩
            gsd_label='drone',       # 또는 '4cm'
            keep_stack=False,        # True면 result_folder/_stack/ 에 VRT 보관 (디버깅용)
        )

    else:
        raise ValueError(f"알 수 없는 MODE: {MODE}. 'wv3' 또는 'drone' 중 선택.")
