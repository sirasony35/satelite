import rasterio
import rasterio.mask
import geopandas as gpd
import os
import glob


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


def batch_crop_pipeline(tif_dir, shp_dir, output_dir):
    """
    result_pan/ 의 두 산출물을 모두 필지 단위로 분할.
      - *_CleanStandardRGB*.tif (uint8 3밴드): HPF 보정된 색감 보존
      - *_PANSHARP_8B*.tif       (uint16 8밴드): 식생지수 산출용 원본 반사율 보존
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_files = sorted(glob.glob(os.path.join(tif_dir, "*_CleanStandardRGB*.tif")))
    pan8b_files = sorted(glob.glob(os.path.join(tif_dir, "*_PANSHARP_8B*.tif")))
    shp_files = sorted(glob.glob(os.path.join(shp_dir, "*.shp")))

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


if __name__ == "__main__":
    input_tif_folder = "wv_data/result_pan"
    input_shp_folder = "wv_data/Shapefile"
    output_crop_folder = "wv_data/crop_result"

    batch_crop_pipeline(input_tif_folder, input_shp_folder, output_crop_folder)
