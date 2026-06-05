"""
result_gaps/ 결과물 일괄 분석 — Otsu 임계, 지수 분포, 결주율, 분포 품질 점검.
사용 지수(NDVI/MSAVI2/NDRE)에 무관하게 동작.
결과는 UTF-8 텍스트 파일로 저장. 콘솔에는 ASCII 요약만.
"""
import os
import sys
import glob
import re
import numpy as np
import rasterio
import geopandas as gpd
from skimage.filters import threshold_otsu

# Windows cp949 콘솔에서도 안전하도록 stdout 강제 UTF-8 (Python 3.7+)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


GAPS_DIR = "wv_data/result_gaps"
REPORT_PATH = "wv_data/result_gaps/_analysis_report.txt"

INDEX_SUFFIXES = ['NDVI', 'MSAVI2', 'NDRE']


def otsu_bimodality_quality(values, threshold):
    """Otsu BCV/TCV — 분포 이중모드 강도. 0~1, 높을수록 신뢰."""
    low = values[values <= threshold]
    high = values[values > threshold]
    if len(low) == 0 or len(high) == 0:
        return 0.0
    n = len(values)
    w0, w1 = len(low) / n, len(high) / n
    mu0, mu1 = low.mean(), high.mean()
    mu = values.mean()
    bcv = w0 * (mu0 - mu) ** 2 + w1 * (mu1 - mu) ** 2
    tcv = values.var()
    return float(bcv / tcv) if tcv > 1e-12 else 0.0


def analyze_one(idx_path, index_name):
    base = os.path.basename(idx_path).replace(f'_{index_name}.tif', '')
    m = re.match(r'(SM\d+)_(\d+)_(\d+)', base)
    if not m:
        return None
    field, date, gsd = m.group(1), m.group(2), m.group(3)

    with rasterio.open(idx_path) as src:
        arr = src.read(1)
        transform = src.transform
        pixel_area = abs(transform[0] * transform[4])
        valid = ~np.isnan(arr) & np.isfinite(arr)
        if not valid.any():
            return None
        v = arr[valid]
        thr = float(threshold_otsu(v))
        quality = otsu_bimodality_quality(v, thr)
        field_area = float(valid.sum() * pixel_area)
        plant_frac = float((v > thr).sum() / v.size)

    shp = os.path.join(GAPS_DIR, f"{base}_GAPS.shp")
    if os.path.exists(shp):
        gdf = gpd.read_file(shp)
        gap_area = float(gdf.geometry.area.sum())
        n_poly = len(gdf)
    else:
        gap_area = 0.0
        n_poly = 0

    gap_ratio = (gap_area / field_area * 100) if field_area > 0 else 0

    return {
        'field': field, 'date': date, 'gsd': gsd, 'index': index_name,
        'field_area': field_area,
        'idx_p10': float(np.percentile(v, 10)),
        'idx_med': float(np.median(v)),
        'idx_p90': float(np.percentile(v, 90)),
        'otsu': thr,
        'quality': quality,
        'plant_frac_raw': plant_frac * 100,
        'gap_area': gap_area,
        'gap_ratio': gap_ratio,
        'n_poly': n_poly,
    }


def discover_index_files():
    """폴더 안의 지수 raster를 모두 탐색해 (지수명, 경로) 리스트로 반환."""
    found = []
    for idx_name in INDEX_SUFFIXES:
        files = sorted(glob.glob(os.path.join(GAPS_DIR, f"*_{idx_name}.tif")))
        for f in files:
            found.append((idx_name, f))
    return found


def emit_report(results, out_lines):
    """결과를 텍스트 라인 리스트에 추가."""
    if not results:
        out_lines.append("분석 가능한 결과물이 없습니다.")
        return

    used_indices = sorted(set(r['index'] for r in results))
    out_lines.append("=" * 100)
    out_lines.append(f"분석 대상: {len(results)}개 파일 / 사용 지수: {', '.join(used_indices)}")
    out_lines.append("=" * 100)

    # ── 시기·지수별 평균 ──
    out_lines.append("")
    out_lines.append(">> 시기·지수별 평균 (필지 평균)")
    header = (f"{'date':<8} {'gsd':>4} {'idx':>7} {'n':>3}  "
              f"{'idx_med':>8} {'otsu':>7} {'qual_Q':>7} {'plant%':>7} "
              f"{'gap%':>7} {'필지㎡':>9} {'결주㎡':>9}")
    out_lines.append(header)
    out_lines.append("-" * len(header))

    by_key = {}
    for r in results:
        key = (r['date'], r['gsd'], r['index'])
        by_key.setdefault(key, []).append(r)

    for (date, gsd, idx_name), rs in sorted(by_key.items()):
        n = len(rs)
        out_lines.append(
            f"{date:<8} {gsd:>4} {idx_name:>7} {n:>3}  "
            f"{np.mean([r['idx_med'] for r in rs]):>8.3f} "
            f"{np.mean([r['otsu'] for r in rs]):>7.3f} "
            f"{np.mean([r['quality'] for r in rs]):>7.3f} "
            f"{np.mean([r['plant_frac_raw'] for r in rs]):>7.1f} "
            f"{np.mean([r['gap_ratio'] for r in rs]):>7.2f} "
            f"{np.mean([r['field_area'] for r in rs]):>9,.0f} "
            f"{np.mean([r['gap_area'] for r in rs]):>9,.0f}"
        )

    # ── 분포 품질 분포 ──
    out_lines.append("")
    out_lines.append(">> 분포 품질 Q 분포 (신뢰도 등급 분포)")
    q_high = sum(1 for r in results if r['quality'] >= 0.5)
    q_mid = sum(1 for r in results if 0.3 <= r['quality'] < 0.5)
    q_low = sum(1 for r in results if r['quality'] < 0.3)
    out_lines.append(f"  강한 이중모드 (Q≥0.5):     {q_high:>4}개  ({q_high/len(results)*100:.1f}%)")
    out_lines.append(f"  적정 이중모드 (0.3~0.5):   {q_mid:>4}개  ({q_mid/len(results)*100:.1f}%)")
    out_lines.append(f"  단봉/약한 분리 (Q<0.3):    {q_low:>4}개  ({q_low/len(results)*100:.1f}%)")

    # ── 의심 케이스 ──
    sorted_by_q = sorted(results, key=lambda x: x['quality'])
    out_lines.append("")
    out_lines.append(">> 분포 품질 최저 10개 (신뢰도 의심 — 휴경/수확후/균질 영상 가능성)")
    for r in sorted_by_q[:10]:
        out_lines.append(
            f"  {r['field']}_{r['date']}_{r['gsd']}_{r['index']} | "
            f"Q={r['quality']:.3f} | idx_med={r['idx_med']:.3f} | "
            f"Otsu={r['otsu']:.3f} | gap={r['gap_ratio']:.2f}% | poly={r['n_poly']}"
        )

    out_lines.append("")
    out_lines.append(">> 분포 품질 최고 10개 (가장 신뢰 가능)")
    for r in sorted_by_q[-10:][::-1]:
        out_lines.append(
            f"  {r['field']}_{r['date']}_{r['gsd']}_{r['index']} | "
            f"Q={r['quality']:.3f} | idx_med={r['idx_med']:.3f} | "
            f"Otsu={r['otsu']:.3f} | gap={r['gap_ratio']:.2f}% | poly={r['n_poly']}"
        )

    # ── 결주율 극단 ──
    out_lines.append("")
    out_lines.append(">> 결주율 분포")
    gaps = [r['gap_ratio'] for r in results]
    out_lines.append(
        f"  min={min(gaps):.2f}%  p25={np.percentile(gaps,25):.2f}%  "
        f"median={np.median(gaps):.2f}%  p75={np.percentile(gaps,75):.2f}%  "
        f"max={max(gaps):.2f}%  mean={np.mean(gaps):.2f}%"
    )

    # 신뢰도 높은 것만 따로
    reliable = [r for r in results if r['quality'] >= 0.3]
    if reliable:
        gaps_r = [r['gap_ratio'] for r in reliable]
        out_lines.append("")
        out_lines.append(f">> 신뢰 가능 케이스만 (Q≥0.3, {len(reliable)}개) 결주율 분포")
        out_lines.append(
            f"  min={min(gaps_r):.2f}%  p25={np.percentile(gaps_r,25):.2f}%  "
            f"median={np.median(gaps_r):.2f}%  p75={np.percentile(gaps_r,75):.2f}%  "
            f"max={max(gaps_r):.2f}%  mean={np.mean(gaps_r):.2f}%"
        )

    # ── 동일 필지·근접 시기 30/50cm 비교 ──
    field_results = {}
    for r in results:
        field_results.setdefault(r['field'], []).append(r)
    pairs = []
    for field, rs in field_results.items():
        # 같은 지수끼리만 비교
        for idx_name in used_indices:
            rs_idx = sorted([r for r in rs if r['index'] == idx_name], key=lambda x: x['date'])
            for i in range(len(rs_idx) - 1):
                a, b = rs_idx[i], rs_idx[i + 1]
                if a['gsd'] != b['gsd'] and abs(int(b['date']) - int(a['date'])) <= 10:
                    pairs.append((field, a, b))

    if pairs:
        out_lines.append("")
        out_lines.append(">> 동일 필지 30cm vs 50cm (인접 시기 ±10일) — 해상도 영향 점검")
        out_lines.append(
            f"  {'field':<6} {'idx':>7} | "
            f"{'30cm date':>10} {'gap%':>7} {'Q':>6} | "
            f"{'50cm date':>10} {'gap%':>7} {'Q':>6} | {'gap_diff':>8}"
        )
        for field, a, b in pairs[:30]:
            a30 = a if a['gsd'] == '30' else b
            b50 = b if b['gsd'] == '50' else a
            diff = b50['gap_ratio'] - a30['gap_ratio']
            out_lines.append(
                f"  {field:<6} {a30['index']:>7} | "
                f"{a30['date']:>10} {a30['gap_ratio']:>6.2f}% {a30['quality']:>6.3f} | "
                f"{b50['date']:>10} {b50['gap_ratio']:>6.2f}% {b50['quality']:>6.3f} | "
                f"{diff:>+7.2f}%"
            )


def main():
    if not os.path.isdir(GAPS_DIR):
        print(f"[중단] 폴더가 없습니다: {GAPS_DIR}")
        return

    idx_files = discover_index_files()
    if not idx_files:
        print(f"[중단] {GAPS_DIR} 에서 지수 raster ({'/'.join(INDEX_SUFFIXES)})를 찾을 수 없습니다.")
        print("       wv3_detecting_gaps.py 를 먼저 실행해주세요.")
        return

    print(f"[Run] {len(idx_files)} index rasters found. Processing...")
    results = []
    for idx_name, idx_path in idx_files:
        r = analyze_one(idx_path, idx_name)
        if r:
            results.append(r)

    out_lines = []
    emit_report(results, out_lines)

    # UTF-8 텍스트 파일로 저장 (cp949 충돌 회피)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines) + '\n')

    # 콘솔에는 ASCII 요약만 (cp949 안전)
    print(f"\n[Done] Report saved: {REPORT_PATH}")
    print(f"       Files analyzed: {len(results)}")
    if results:
        used = sorted(set(r['index'] for r in results))
        print(f"       Indices used: {', '.join(used)}")
        q_low = sum(1 for r in results if r['quality'] < 0.3)
        q_mid = sum(1 for r in results if 0.3 <= r['quality'] < 0.5)
        q_high = sum(1 for r in results if r['quality'] >= 0.5)
        print(f"       Quality Q:  high(>=0.5) {q_high} / mid(0.3-0.5) {q_mid} / low(<0.3) {q_low}")
        gaps = [r['gap_ratio'] for r in results]
        print(f"       gap_ratio:  min={min(gaps):.1f}%  median={np.median(gaps):.1f}%  max={max(gaps):.1f}%")
    print(f"\n  Detailed report: open the txt file above in a text editor.")


if __name__ == "__main__":
    main()
