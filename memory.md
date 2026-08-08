# 프로젝트 개요

WorldView-3(WV3) 위성 다중분광 영상을 처리하여 **농경지(논콩) 작물 모니터링용 처방지도**를 생산하는 원격탐사 파이프라인입니다. 대상 지역은 새만금 간척지(필지 코드 `SM01`, `SM02` 등)이며, 30cm / 50cm 해상도(GSD)의 여러 시기 영상을 다룹니다.

핵심 처리 흐름은 다음과 같습니다.

1. **팬샤프닝(Pansharpening)** — MUL(8밴드 다중분광) + PAN(전정색)을 합성해 고해상 8밴드 영상과 시각화용 RGB를 생성하고 EPSG:5179로 재투영.
2. **필지 분할(Crop)** — 필지 경계 Shapefile로 영상을 잘라냄.
3. **식생지수/결주 탐지** — MSAVI2·NDVI·NDRE 등 식생지수를 계산하고 Otsu 임계로 작물/토양을 분리, 결주(작물이 빠진 구역)를 폴리곤으로 추출.
4. **활력도(VRA) 분석** — Green-Yellow 기반 활력 지수로 영양 결핍 구역(하위 n%)을 추출.
5. **결과 일괄 분석/품질 점검** — 결주율·분포 신뢰도(Q)를 종합 리포트로 정리.

`.py` 파일들이 정식 파이프라인이고, `.ipynb` 노트북들은 초기 실험/프로토타입(방사보정·Brovey 팬샤프닝 시험)입니다.

---

# 폴더/파일 구조

```
satelite/
├─ memory.md                     # (본 문서)
├─ wv3_rgb_pansharpening.py      # [1] 팬샤프닝 (8B + RGB 듀얼 출력, EPSG:5179 재투영)
├─ wv3_spatial_crop.py           # [2] 필지 SHP로 분할 (WV3 듀얼출력 / PlanetScope / 폴리곤별 개별 분할)
├─ wv3_scientific_vi_crop.py     # [2'] 식생지수 TIF를 VRT 재투영 후 정밀 분할
├─ wv3_detecting_gaps.py         # [3] 결주 탐지 본체 (다중센서: WV3/드론/PlanetScope, 식생지수→Otsu→폴리곤)
├─ _analyze_gaps.py              # [4] result_gaps 결과물 일괄 품질 분석 리포트
├─ wv3_eci_calculator.py         # [+] 활력도(VRA) 처방지도 산출 (Green-Yellow 지수)
│
├─ Panshaperned.ipynb            # (실험) 방사보정 + 팬샤프닝 프로토타입
├─ pan_mul_combine.ipynb         # (실험) Brovey 팬샤프닝 + 반사율 보정 프로토타입
├─ data_mul.ipynb                # (실험) 원본 WV3 XML/TIF 탐색·밴드 매칭 (대용량)
│
├─ wv_analysis.qgz               # QGIS 프로젝트 파일
├─ .idea/                        # PyCharm 설정
└─ wv_data/                      # 데이터 루트
   ├─ mul_data/      # 입력: SM_<seq>_<date>_<gsd>_MUL/PAN_<n>.tif (정규화된 MUL/PAN 쌍)
   ├─ ShapeFile/     # 필지 경계 폴리곤 (SM01.shp, SM02.shp ...)
   ├─ result_pan/    # 팬샤프닝 산출 (*_PANSHARP_8B*.tif, *_CleanStandardRGB*.tif)
   ├─ crop_result/   # 필지 분할된 8밴드·RGB (*_PANSHARP_8B_Crop.tif 등)
   ├─ vi_data/       # 식생지수 raster (분할 전)
   ├─ vi_crop/       # 식생지수 필지 분할 결과
   ├─ result_gaps/   # 결주 탐지 산출 (*_GAPS.shp/.png, *_MSAVI2.tif, _analysis_report.txt)
   └─ result/        # 기타 결과
```

> 참고: 코드 내 경로는 모두 상대경로(`wv_data/...`)이므로 스크립트는 `satelite/` 폴더를 작업 디렉터리로 두고 실행해야 합니다. 일부 코드에서 `wv_data/Shapefile`(소문자 f)로 참조하나 실제 폴더는 `ShapeFile`이라 Windows 외 환경에서는 대소문자 주의가 필요합니다.

---

# 주요 스크립트 상세

## wv3_rgb_pansharpening.py — 팬샤프닝 (듀얼 출력)
- **목적**: WV3 PAN 응답대역(450~800nm)에 맞춘 Pseudo-PAN 가중치로 팬샤프닝하고, 동시에 두 가지 산출물을 만든 뒤 EPSG:5179로 재투영.
  - 분석용 8밴드(uint16, 원본 수치 보존) — 비율(ratio) 기반 팬샤프닝.
  - 시각화용 RGB(uint8) — **HPF(High-Pass Filter) 디테일 주입** 방식. PAN 고주파 성분(`PAN - box_blur(PAN)`)을 밴드별 게인으로 가산해 DC(평균 색상)를 보존 → QGIS 색감과 일치. (IHS/Brovey의 곱셈 ratio가 식생 밀집 타일에서 색감을 왜곡하는 문제를 회피)
  - 유효 footprint 마스크(PAN·MUL 모두 유효)로 경계 outlier 차단, p2/p98 스트레칭.
  - `_box_blur`: scipy 없이 numpy cumsum만으로 박스 평활(저주파 추출).
  - `PAN_WEIGHTS_WV3`: Coastal(B1)·NIR2(B8)는 기여 0, 가시·근적외 5밴드 합=1.
- **입력**: `wv_data/mul_data/` 의 `*MUL*.tif` 와 동명의 `*PAN*.tif` 쌍.
- **출력**: `wv_data/result_pan/` 에 `*_PANSHARP_8B*.tif`(8밴드 uint16), `*_CleanStandardRGB*.tif`(3밴드 uint8 RGB).
- **진입점**: `process_batch_pipeline("wv_data/mul_data", "wv_data/result_pan")`.

## wv3_spatial_crop.py — 필지 분할 (다중 소스 대응)
- **목적**: raster를 필지 경계 SHP로 마스킹/crop. 어느 경로든 **추가 스트레칭·스케일링 없이 원본 dtype·값 그대로** 저장(팬샤프닝/원본 단계에서 이미 색감·반사율 확정). (기존 파일명 `wv3_spatial_corp.py`의 `corp` 오타를 `crop`으로 정정하며 재작성됨)
- **공통 함수 `crop_field(src, shp, out)`**: SHP를 raster CRS로 자동 변환 후 `rasterio.mask.mask(crop=True)`로 잘라 LZW 압축 GTiff로 저장. 세 파이프라인 모두 이 함수를 공유.
- **세 가지 처리 모드 (`__main__` MODE 스위치: `'wv3'`/`'planetscope'`/`'pcc'`/`'pcc_rgb'`)**:
  - **`batch_crop_pipeline()` [WV3]**: `result_pan/`의 `*_CleanStandardRGB*.tif`(uint8 3밴드)와 `*_PANSHARP_8B*.tif`(uint16 8밴드)를 모든 SHP로 분할. 파일명 `SM_<seq>_<date>_<gsd>_<type>_<n>`을 파싱해 `{필지}_{seq}_{date}_{gsd}_{type}_Crop.tif` 생성.
  - **`crop_polygons_individually()` [PCC/RGB]**: **하나의** multi-band raster를 **하나의 SHP에 든 여러 폴리곤별로 개별** TIF로 분할(필지당 SHP가 별도가 아닌 경우용). `name_field`(UID/ID/PNU 등) 컬럼으로 파일명 생성, 없으면 `poly000…` 인덱스. `bands=[6,4,2]`처럼 밴드 선택 가능하며 정확히 3밴드면 RGB ColorInterp/photometric을 설정해 QGIS 트루컬러 렌더. `skip_existing`으로 기존 출력 스킵, raster 범위 밖 폴리곤은 자동 스킵.
  - **`run_planetscope_batch()` [PlanetScope]**: SuperDove 8개 single-band TIF(`{base}_{BAND}.tif`)를 `discovery_band`(기본 'Red') 기준으로 scene 식별 → `wv3_detecting_gaps.stack_drone_bands`를 재사용해 8밴드로 스택 → 모든 SHP로 분할. 출력 `{field}_{base}_{gsd_label}_PS_8B_Crop.tif`(이후 `SENSOR_PRESETS['planetscope_superdove']`로 결주 분석).
- **입력**: 모드별 — `wv_data/result_pan/`(WV3) / `wv_data/planetscope_data/`(PS) / 단일 TIF+SHP(PCC) + 각 `*.shp`.
- **출력**: `wv_data/crop_result/`(WV3) / `wv_data/planetscope_crop_result/`(PS) / `pcc/crop_result(_rgb)/`(PCC).

## wv3_scientific_vi_crop.py — 식생지수 정밀 분할 (VRT)
- **목적**: 식생지수 TIF를 자르되, `WarpedVRT`로 **메모리상에서 먼저 EPSG:5179로 재투영한 뒤** 5179 좌표계 폴리곤으로 잘라 외곽 NoData 테두리 왜곡을 원천 차단. 식생지수 수치 보존을 위해 Bilinear 사용, float은 NaN / 정수는 -9999를 NoData로.
- **입력**: `wv_data/vi_data/*.tif` + `wv_data/Shapefile/*.shp`.
- **출력**: `wv_data/vi_crop/` 에 `{필지}_{seq}_{date}_{gsd}_{index}_Crop.tif`.
- **진입점**: `batch_scientific_vi_pipeline("wv_data/vi_data", "wv_data/Shapefile", "wv_data/vi_crop")`.

## wv3_detecting_gaps.py — 결주 탐지 본체 (핵심)
- **목적**: 다중분광 raster에서 식생지수를 계산하고 결주(작물 빠진 구역)를 폴리곤으로 추출. WV3 위성뿐 아니라 MicaSense·DJI(드론)·PlanetScope 등 센서별 밴드 프리셋(`SENSOR_PRESETS`)으로 범용 동작.
  - **센서 프리셋**: `SENSOR_PRESETS` dict — `wv3`/`micasense_rededge`/`micasense_altum`/`dji_phantom4_ms`/`dji_mavic3m`/`sentera_6x`/`parrot_sequoia`/`planetscope_superdove` 각각의 (Red, NIR, RedEdge) 1-based 인덱스 정의. 새 센서는 이 인덱스만 추가하면 됨.
  - **식생지수**: `compute_vegetation_index()` — `msavi2`(기본, 간척지 사질토 권장)/`ndvi`/`ndre` 선택. 밴드 인덱스를 인자로 노출(WV3 기본: Red=5, NIR=7, RedEdge=6). 입력 밴드 수 부족 시 명시 에러.
  - **임계/신뢰도**: `threshold_otsu`로 작물/토양 분리, `otsu_bimodality_quality()`로 분포 이중모드 강도 Q(0~1) 산출(Q≥0.5 강함 / 0.3~0.5 적정 / <0.3 의심). 절대 NDVI 값에 무관해 식생 NDVI가 낮은 간척지에 적합.
  - **형태 처리**: 행간(`row_spacing_m`, 논콩 0.65m)·픽셀 크기로 `closing_radius` 자동 산출(`round(행간/픽셀/2)`, 범위 [1,10]) 후 morphological closing, `min_gap_area_sqm`(0.5㎡) 미만 객체 제거.
  - **벡터화/시각화**: 결주 마스크를 폴리곤화(`area_sqm`, `otsu_qual` 속성), simplify 후 SHP 저장. `render_gap_png()`로 RGB 배경 위 결주 오버레이 PNG(matplotlib 미설치 시 PNG만 스킵). 한글 폰트 직접 등록.
  - **드론 single-band 처리**: `stack_drone_bands()` — 드론의 밴드별 single-band TIF(`{base}_{BAND}.tif`)를 묶어 multi-band raster로 합침. osgeo 있으면 GDAL VRT(가상 raster, 디스크 절약), 없으면 multi-band GeoTIFF 폴백(`DRONE_STACK_EXT`). `run_drone_batch()`가 `*_RED.tif` 기준으로 필지 식별→스택→`{base}_RGB.tif` 배경 자동 인식→`detect_missing_plants` 호출, 처리 후 임시 스택 정리(`keep_stack`). 이 스택 함수는 `wv3_spatial_crop.py`의 PlanetScope 배치에서도 재사용됨.
  - **배치/실행**: `run_batch()`(폴더 패턴 일괄, `rgb_resolver` 콜백 지원) 재사용 함수 + `run_drone_batch()`. `MODE` 스위치로 `'wv3'`(위성) / `'drone'`(single-band 묶음) / `'planetscope'`(SuperDove 8밴드 분할 결과) 선택. (현재 파일 기본값은 `MODE='drone'`)
- **입력**: 모드별 — `crop_result/*_PANSHARP_8B_Crop.tif`(WV3) / `drone_data/`의 밴드별 single-band TIF 세트(드론) / `planetscope_crop_result/*_PS_8B_Crop.tif`(PlanetScope). WV3 PNG 배경 RGB는 파일명 `PANSHARP_8B_Crop`→`CleanStandardRGB_Crop` 치환으로 자동 탐색.
- **출력**: 결과 폴더(`result_gaps`/`drone_result_gaps`/`planetscope_result_gaps`)에 `{필지}_{date}_{gsd}_GAPS.shp`(결주 폴리곤), `..._{INDEX}.tif`(식생지수 raster), `..._GAPS.png`(오버레이).
- **진입점**: `__main__`의 `MODE` 분기에서 `run_batch(...)` 또는 `run_drone_batch(...)` 호출.

## _analyze_gaps.py — 결주 결과 일괄 품질 분석
- **목적**: `result_gaps/` 의 모든 식생지수 raster + GAPS SHP를 종합 분석해 텍스트 리포트 생성. Otsu 임계, 지수 분포(p10/median/p90), 결주율, 분포 품질 Q, 신뢰도 등급 분포, 의심/신뢰 상위·하위 10개, 동일 필지 30cm vs 50cm(±10일) 해상도 영향 비교 등을 정리. 사용 지수(NDVI/MSAVI2/NDRE) 무관 동작. cp949 콘솔 충돌 회피를 위해 리포트는 UTF-8 파일로, 콘솔에는 ASCII 요약만 출력.
- **입력**: `wv_data/result_gaps/*_{NDVI|MSAVI2|NDRE}.tif` + 동명 `*_GAPS.shp`.
- **출력**: `wv_data/result_gaps/_analysis_report.txt`.
- **진입점**: `main()` (`python _analyze_gaps.py`).

## wv3_eci_calculator.py — 활력도(VRA) 처방지도 산출
- **목적**: WV 다중분광 영상으로 **Normalized Green-Yellow 활력 지수** `(Green - Yellow)/(Green + Yellow)`를 산출하고, 하위 n%(`stress_percentile`, 기본 15%) 영양 결핍 구역을 Shapefile로 추출. 생육 단계 토글(`growth_stage`): `late`는 NDVI Otsu로 식생만 엄격 분리, `early`는 토양 포함 분석. 밴드(3=Green, 4=Yellow, 5=Red, 7=NIR1).
- **입력**: `wv_data/mul_data/SM_01_250728_30_MUL_1.tif` 같은 MUL 영상(테스트 하드코딩).
- **출력**: 입력과 같은 폴더에 `{필지}_{date}_VIGOR.tif`(활력 지수), `{필지}_{date}_VRA.shp`(결핍 처방지도).
- **진입점**: `generate_vra_map(...)` (`__main__` 테스트).

## 노트북 (실험/프로토타입)
- **data_mul.ipynb**(대용량): 원본 WV3 납품 데이터(`A지역`/`B지역`, 30cm/50cm)의 XML(`ABSCALFACTOR`, `EFFECTIVEBANDWIDTH`, `MEANSUNEL`) 파싱과 GeoTIFF 밴드 매칭·탐색.
- **Panshaperned.ipynb / pan_mul_combine.ipynb**: 방사보정(DN→반사율, 태양 고도각 보정) + 팬샤프닝(Brovey 변환 등) 초기 프로토타입. 정식 `.py` 팬샤프닝의 전신.

---

# 데이터 흐름 / 실행 순서

```
[원본 WV3 납품 데이터: MUL + PAN + XML]
        │  (노트북에서 탐색·방사보정 실험 → 정규화된 SM_<seq>_<date>_<gsd>_MUL/PAN_<n>.tif 준비)
        ▼
wv_data/mul_data/
        │  ① wv3_rgb_pansharpening.py
        ▼
wv_data/result_pan/   (PANSHARP_8B + CleanStandardRGB)
        │  ② wv3_spatial_crop.py   (MODE='wv3'; + ShapeFile 필지 경계)
        ▼
wv_data/crop_result/  (*_PANSHARP_8B_Crop.tif, *_CleanStandardRGB_Crop.tif)
        │  ③ wv3_detecting_gaps.py  (MODE='wv3', MSAVI2 + Otsu)
        ▼
wv_data/result_gaps/  (*_GAPS.shp/.png, *_MSAVI2.tif)
        │  ④ _analyze_gaps.py
        ▼
wv_data/result_gaps/_analysis_report.txt  (결주율·신뢰도 종합 리포트)
```

부가 경로(활력도/식생지수 분할):
- `wv3_eci_calculator.py`: `mul_data/` MUL → `VIGOR.tif` + `VRA.shp` (영양 결핍 처방지도).
- `wv3_scientific_vi_crop.py`: `vi_data/` 식생지수 TIF → `vi_crop/` (VRT 재투영 후 정밀 분할).

**표준 실행 순서**: ① 팬샤프닝 → ② 필지 분할 → ③ 결주 탐지 → ④ 결과 분석. (③ 실행 전 ②까지 완료 필요; ④ 실행 전 ③ 필요.)

---

# 의존성

- **Python 3.7+** (`sys.stdout.reconfigure` 사용)
- 핵심 라이브러리:
  - `rasterio` — raster I/O, 재투영(`warp`), 마스킹(`mask`), `WarpedVRT`, `features.shapes`
  - `geopandas` — Shapefile 입출력, 폴리곤 면적/CRS 변환
  - `numpy` — 배열 연산
  - `scikit-image`(`skimage`) — `filters.threshold_otsu`, `morphology`(remove_small_objects, closing, disk)
- 선택(있으면 시각화 단계 동작, 없으면 자동 스킵):
  - `matplotlib` — 결주 오버레이 PNG (`wv3_detecting_gaps.py`)
- 좌표계: 산출물은 **EPSG:5179**(Korea 2000 / Unified CS) 기준.
- 한글 폰트: PNG용 맑은 고딕(`C:\Windows\Fonts\malgun.ttf`) 등 직접 등록.
- 외부 도구: **QGIS**(`wv_analysis.qgz`)로 결과 검증/시각화.

설치 예시:
```
pip install rasterio geopandas numpy scikit-image matplotlib
```
(Windows에서는 GDAL/rasterio/geopandas 바이너리 의존성 때문에 conda 환경 권장)

---

# 비고

- **파일명 규약**이 파이프라인 전반의 메타 파싱에 핵심입니다.
  - 영상: `SM_<seq>_<date(YYMMDD)>_<gsd(30|50)>_<TYPE>_<n>.tif` (예: `SM_01_250728_30_MUL_1.tif`).
  - `TYPE`: `MUL`/`PAN`(원본) → `PANSHARP_8B`/`CleanStandardRGB`(팬샤프닝) → `..._Crop`(분할). 필지 코드 `SM01`/`SM02`는 ShapeFile 이름에서 옴.
  - `wv3_spatial_crop.py`/`wv3_scientific_vi_crop.py`/`_analyze_gaps.py`/`wv3_detecting_gaps.py`가 `_` 분할로 seq/date/gsd/type을 추출하므로 명명 규칙을 어기면 스킵·오류 발생.
- **WV3 8밴드 순서**: 1 Coastal / 2 Blue / 3 Green / 4 Yellow / 5 Red / 6 RedEdge / 7 NIR1 / 8 NIR2.
- **간척지 특화**: 식생 NDVI가 본래 낮은 새만금 환경 때문에, 절대 임계 대신 Otsu + 분포 품질 Q와 MSAVI2(토양 보정)를 채택. Q<0.3이면 휴경/수확후/균질 영상으로 신뢰도 낮음으로 표시.
- **인코딩 주의**: 노트북·일부 스크립트는 한글 주석 포함. `_analyze_gaps.py`는 cp949 콘솔 충돌을 피하려 리포트를 UTF-8 파일로 저장하고 콘솔엔 ASCII만 출력. 노트북 일부 셀의 주석은 인코딩이 깨진 채 저장돼 있음(기능엔 무관).
- **경로 하드코딩**: 모든 입출력 경로가 `wv_data/...` 상대경로이며 일부는 테스트 파일이 하드코딩되어 있어, 다른 데이터로 돌릴 때는 `__main__` 블록 또는 진입 함수 인자를 수정해야 함.
- **대용량 파일**: `data_mul.ipynb`(약 5MB) 및 `wv_data/` 내 GeoTIFF는 용량이 크므로, git 공유 시 `.gitignore`로 `wv_data/`(특히 산출물)와 대용량 출력을 제외하는 것을 권장.

---

# 현재 상태 및 진행 계획 (2026-08-08 갱신)

## 데이터 현황 (wv_data/)
- **mul_data**: **원본 납품 형식(WorldView Legion) 6개 장면** — 밀_보리 4장면(`SM_03_260312_30`, `SM_05_260403_50`, `SM_06_260416_50`, `SM_07_260504_50`) + 콩 2장면(`RL_01_260702_50`, `SM_01_260710_50`). 상세 구조는 아래 "WorldView Legion 원본 납품 데이터" 섹션. 구 정규화 파일(2025 시기 5개 포함)은 삭제된 상태.
- **ShapeFile**: 필지 경계 `SM01`~`SM24` **24개** SHP + **GJSM-1-1/1-2/1-3/2-2/2-3 5개**(RL 지역 필지, 2026-08-08 zip 입수·추출. 2-1은 없음). GJSM 원본 zip(`GJSM-*_Boundary.zip`)도 같은 폴더에 보존. GJSM은 EPSG:4326, 필지당 1폴리곤(~1.6-1.8ha), 컬럼은 드로잉툴 산출(color/fill/name 등). 파일명은 후속 `_` 파싱 보호를 위해 `_Boundary` 접미사 제거하고 하이픈만 사용.
- **result_pan**: **11개 장면** 팬샤프닝 산출물(`PANSHARP_8B` + `CleanStandardRGB`, 각 `_1` 접미사) —
  - 2026 Legion 6장면(밀_보리 4 재생성 + 콩 2 신규): **Legion 가중치로 2026-08-08 생성** (최신·정합).
  - 2025 구 5시기(250728/250731/250820/250829/251028): 구 WV3 가중치 산출물 그대로(원본 소실로 재처리 불가. 단 ratio 방식 특성상 정규화 지수엔 영향 없음).
  - `result/` 폴더에 구 산출물 7.2GB 중복 보관 중(정리 후보).
- **crop_result**: 파이프라인 Crop **250개**(콩 58 = SM 48 + GJSM 10 / 밀_보리 192 = 4장면×24필지×8B·RGB) + 외부 생성 `VisualLocked` 216개 공존. `VisualLocked`는 어느 스크립트에도 없는 명명(QGIS 시각화 고정본 추정).
- **vi_crop**: 식생지수 raster **500개** — 필지 8B Crop당 NDVI/GNDVI/NDRE/MSAVI2 4종 (`wv_vi_extract.py`, preset='wv_legion'). vi_data는 비어 있음.
- **result_gaps**: `SM01_250728_GAPS.shp`, `SM_250728_GAPS.shp` 2세트만 존재 — 결주 탐지는 구 데이터 일부만 돌린 상태(신규 Legion 장면 미실행).
- **PlanetScope(PSS) 데이터는 아직 로컬에 없음**: `wv_data/planetscope_data/` 미존재, `pcc/` 폴더도 미존재. 밴드 순서는 사용자 확정(1~8 = CoastalBlue/Blue/GreenI/Green/Yellow/Red/RedEdge/NIR = 코드 preset과 일치).

## 진행 상황 (2026-08-08 기준)
**목표: WV·PSS 위성 데이터의 "필지별 분할 → 식생지수 추출" (2026-08-05 사용자 지시)**

- ✅ **WV(Legion) 경로 완료**: 입고 6개 장면 전부 팬샤프닝(Legion 가중치) → 필지 분할 → 4종 식생지수 추출까지 완료. 콩 시계열: RL 5필지 NDVI median 0.05~0.10(7/2), SM 24필지 -0.05~0.17(7/10, 파종 초기). 밀_보리 시계열: 0.12~0.18(3/12) → 0.3~0.77(5/4, 최성기; SM18은 0.17로 유독 낮음).
- ⏳ **PSS 경로 대기**: 데이터 입수 시 형태별 분기 —
  - (a) 밴드별 single-band TIF(`{base}_{BAND}.tif`)면 `wv3_spatial_crop.py` MODE=`'planetscope'` → `wv_vi_extract.py` MODE=`'planetscope'`.
  - (b) 단일 8밴드 scene(Planet `AnalyticMS_SR_8b` 등)이면 `crop_polygons_individually()` 활용 또는 batch 변형.
  - 3m GSD라 결주 탐지 시 `min_gap_area_sqm=2.0`, `closing_radius=1` 권장.
- 이후 후보: 신규 장면 결주 탐지(`wv3_detecting_gaps.py`, preset='wv_legion' 사용), 활력도(VRA) 산출, `result/` 중복 7.2GB 정리.

---

# WorldView Legion 원본 납품 데이터 (2026-08-08 입고·학습)

`wv_data/mul_data/`에 원본 납품 형식 그대로 6개 장면 입고. **위성은 WV3가 아니라 WorldView Legion**(SATID: LG01/LG02/LG03/LG06)이며, 밴드 구성이 기존 코드 가정과 다름 — 아래 "코드 영향" 필독.

## 폴더 구조
```
mul_data/
├─ 밀_보리/<납품일>/<30cm|50cm>/     # 납품일 폴더명(260316 등)은 촬영일(260312)과 다를 수 있음
│   └─ {SCENE}_BAND / _GIS_FILES / _MUL / _PAN   # 장면당 4개 폴더
└─ 콩/<납품일>/...                    # 콩 260710은 GSD 중간 폴더 없이 바로 장면 폴더
```
- SCENE 명명: `{지역}_{seq}_{촬영일YYMMDD}_{gsd}` — 지역코드 SM(새만금) 외 **RL 신규 등장**(콩 260702).
- **장면 인벤토리 (6개)**:
  - 밀_보리: `SM_03_260312_30`(LG02), `SM_05_260403_50`(LG01), `SM_06_260416_50`(LG01), `SM_07_260504_50`(LG03) — 기존 result_pan에 팬샤프닝 결과가 이미 있는 시기들의 원본.
  - 콩: `RL_01_260702_50`(LG06, **신규 지역**), `SM_01_260710_50`(LG03, **신규 시기**) — 미처리 신규 작업 대상.

## 장면당 4개 폴더 내용
- **_MUL/**: `{SCENE}_MUL.tif` (8밴드 uint16 DN, nodata=0, EPSG:32652 UTM 52N) + `MUL.XML`/`MUL.IMD`(방사보정 계수: 밴드별 `ABSCALFACTOR`·`EFFECTIVEBANDWIDTH`, `MEANSUNEL`·`MEANSUNAZ`, `CLOUDCOVER` 등) + `MUL.RPB`(RPC) + `MUL.TIL` + `BROWSE.JPG` + LICENSE/README. 제품레벨 LV2A(ORStandard2A), Radiometric=Corrected.
- **_PAN/**: `{SCENE}_PAN.tif` (1밴드 uint16) + 동일 메타 세트.
- **_BAND/**: **벤더 제공 식생지수** `{SCENE}_NDVI/NDRE/GNDVI.tif` (float32, MUL 해상도, nodata=-3.4e38) — `vi_data/`→`wv3_scientific_vi_crop.py` 경로에 바로 쓸 수 있는 후보. 단 벤더가 어떤 밴드로 계산했는지는 메타에 없음.
- **_GIS_FILES/**: footprint SHP 5종 (ORDER/PRODUCT/STRIP/TILE/MUL·PAN PIXEL_SHAPE) — 필지 경계 아님, 영상 커버리지용.

## 해상도 사양
- "50cm 제품" = PAN 0.5m + MUL 2.0m / "30cm 제품" = PAN 0.3m + MUL 1.2m (PAN:MUL = 1:4).

## ⚠️ 코드 영향 — Legion 밴드 구성이 WV3와 다름 (핵심)
- **Legion 8밴드**: 1 Coastal / 2 Blue / 3 Green / 4 Yellow / 5 Red / **6 RedEdge1 / 7 RedEdge2 / 8 NIR** (XML `BAND_C/B/G/Y/R/RE1/RE2/N` 순).
- WV3 가정(…6 RedEdge / 7 NIR1 / 8 NIR2)과 6~8번이 다름:
  - `SENSOR_PRESETS['wv3']`(red=5, nir=7, re=6)를 Legion에 쓰면 **NIR 자리에 RedEdge2(B7)가 들어가 NDVI/MSAVI2 왜곡** → Legion용 preset `{red=5, nir=8, red_edge=6}` 신설 필요.
  - `wv3_eci_calculator.py`의 NIR=7 하드코딩도 동일 문제(B8로 수정).
  - `PAN_WEIGHTS_WV3`는 "B8=NIR2 기여 0" 가정 → Legion은 B8이 유일한 NIR이므로 가중치 재검토 필요(B7 RE2/B8 NIR 배분).
  - 기존 result_pan의 2026 시기 산출물은 Legion 원본을 WV3 가정으로 처리한 결과일 가능성 → 팬샤프닝 색감(RGB는 B5/B3/B2라 영향 미미)보다 **8밴드 기반 지수 계산 시 주의**. 2025 시기(250728~251028) 원본은 현재 없어 위성 미확인.
- **파이프라인 어댑테이션 필요**: `process_batch_pipeline()`은 mul_data 평면 폴더에서 `*MUL*.tif`를 glob하고 같은 폴더의 `MUL→PAN` 치환 파일을 찾음 → 새 구조는 (a) 재귀 탐색 + _MUL/_PAN 폴더 대응으로 수정하거나 (b) 평면 폴더로 정규화 복사 필요. 정규화 시 기존 규약대로 **`_1` 접미사 유지 권장**(`SM_01_260710_50_MUL_1.tif`) — crop 단계의 `parts[4:-1]` 타입 파싱이 마지막 `_n`을 전제로 함.

---

# 변경 이력

- **2026-08-08 밀_보리 4장면 Legion 재처리 완료** — 팬샤프닝(Legion 가중치, result_pan 덮어쓰기) → 필지 분할(4장면×24필지×8B/RGB=192개) → 식생지수(384개) 전부 재생성.
  - 이슈: 밀_보리 원본 tif는 `_1` 접미사(`SM_03_260312_30_MUL_1.tif`)가 붙어 있어 초기 `*_MUL.tif` glob이 0건 매칭(콩 장면은 접미사 없음) → `process_raw_delivery_pipeline`을 **`{SCENE}_MUL` 폴더명 기준 탐색**으로 수정(내부 tif는 `*MUL*.tif`/`*PAN*.tif` 글롭). **같은 납품이라도 파일명 접미사가 작물별로 다를 수 있음** — 신규 납품 시 주의.
  - NDVI median은 구 가중치 산출물과 동일(예: SM01_07 0.691) — ratio 팬샤프닝은 픽셀별 동일 배율이라 정규화 지수에서 가중치가 소거되기 때문(수학적으로 예상된 결과). 가중치 수정의 실효는 8B raster의 공간 디테일 충실도.
  - 누적 산출물: crop_result 파이프라인 Crop 250개(콩 58 + 밀_보리 192), vi_crop 지수 500개(콩 116 + 밀_보리 384). 이로써 **입고된 Legion 6개 장면 전부 필지 분할·식생지수까지 완료**. 남은 것: PSS 데이터 입수 대기.

- **2026-08-08 Legion 대응 + 콩 장면 처리** — 밴드 검증: SM20 필지에서 벤더 NDVI median=0.171 vs 산출 NDVI(NIR=B8) median=0.171 정확히 일치 (구 WV3 가정 NIR=B7이면 0.130으로 왜곡). **벤더도 NIR=B8 사용 확인.**
  - `wv3_detecting_gaps.py`: `SENSOR_PRESETS['wv_legion']` = {red=5, nir=8, red_edge=6} 추가.
  - `wv3_rgb_pansharpening.py`: `PAN_WEIGHTS_LEGION`(유효대역폭 비례) 추가, `wv3_pansharpen_dual_export_5179`에 `pan_weights` 인자 추가(기본 Legion), 원본 납품 중첩 구조를 재귀 탐색하는 `process_raw_delivery_pipeline(input_root, output_dir, pan_weights, scene_filter, skip_existing)` 신설, `__main__` MODE=`'raw'`/`'flat'` 스위치.
  - `wv3_spatial_crop.py`: `batch_crop_pipeline`에 `scene_filter` 인자 추가.
  - `wv3_eci_calculator.py`: `band_nir` 인자화(기본 8=Legion, WV3는 7 지정).
  - **`wv_vi_extract.py` 신규**: 8밴드 Crop에서 NDVI/GNDVI/NDRE/MSAVI2만 추출하는 경량 배치(결주 탐지와 분리). `BAND_PRESETS`(wv_legion/wv3/planetscope_superdove), 출력 `{필지}_{seq}_{date}_{gsd}_{INDEX}_Crop.tif` → `vi_crop/`. MODE=`'wv'`/`'planetscope'`.
  - 처리 실행: 콩 2장면 팬샤프닝(Legion 가중치) → SM_01_260710 필지 분할 48개 → 식생지수 96개. 실행 환경: conda `python312` (`C:\Users\Sangho\anaconda3\envs\python312\python.exe`), 작업 디렉터리 `satelite/`.
  - RL_01_260702는 당초 필지 SHP가 없어 분할 0건이었으나, 같은 날 **GJSM 경계 zip 5개 입수 → 추출 후 분할 10개(8B+RGB×5필지) + 식생지수 20개 완료** (NDVI median 0.05~0.10, 7/2 파종 초기 콩). 이로써 콩 2장면은 필지 분할·식생지수까지 전부 완료.

- **2026-06-30 pull 반영** — `wv3_spatial_corp.py` → `wv3_spatial_crop.py` 파일명 정정 및 재작성, `wv3_detecting_gaps.py` 대폭 수정.
  - `wv3_spatial_crop.py`: 기존 WV3 듀얼출력 분할(`batch_crop_pipeline`)에 더해 단일 raster를 SHP 내 폴리곤별로 개별 분할하는 `crop_polygons_individually`(밴드 선택·RGB 추출·ColorInterp 포함)와 PlanetScope SuperDove 8밴드 스택+분할 `run_planetscope_batch`를 추가, `__main__` MODE 스위치를 `wv3`/`planetscope`/`pcc`/`pcc_rgb`로 확장.
  - `wv3_detecting_gaps.py`: 단일 WV3 처리에서 다중 센서 파이프라인으로 확장 — 센서별 밴드 프리셋 `SENSOR_PRESETS`, 드론 single-band TIF 스택(`stack_drone_bands`, VRT/GeoTIFF 폴백)과 `run_drone_batch`, PlanetScope 모드를 추가하고 `MODE` 스위치를 `wv3`/`drone`/`planetscope`로 확장(파일 기본값은 `drone`).
