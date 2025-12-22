# -*- coding: utf-8 -*-
Original file is located at
    https://colab.research.google.com/drive/1CXqIlIIv2wRSgS_KKhWjy22FIhrFDsY1
"""

# ==== one-time installs====
!pip -q install geopandas shapely fiona pyproj rtree pyogrio pandas tqdm pvlib

# ==== imports ====
import os, math, shutil, subprocess, warnings, time
import numpy as np
import pandas as pd
import geopandas as gpd
import pvlib
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points, transform as shp_transform
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
t0 = time.time()

# =========================
# USER CONFIG
# =========================
# Data
STRUCT_ZIP  = "/content/drive/MyDrive/DL/Shadow/DhakaStr.zip"   # all structures
TARGET_ZIP  = "/content/drive/MyDrive/DL/Shadow/all_build_F.shp.zip"  # target buildings
STRUCT_DIR  = "/content/DhakaStr"
TARGET_DIR  = "/content/all_Build_F"
OUT_DIR     = "/content/roofshade_gridray_50m"
ZIP_BASE    = "/content/roofshade_gridray_50m"   # .zip appended

# Attributes
HEIGHT_FIELD   = "Floor"    # floors → meters via FLOOR_HEIGHT_M
FLOOR_HEIGHT_M = 3.0
KEEP_USE_FIELD = "Str_Use1" # keep this in outputs

# Geometry / CRS
EPSG_METERS     = 32646     # WGS84 / UTM 46N (Dhaka, meters)
BUFFER_RADIUS_M = 50.0
GRID_SPACING_M  = 2.0       # grid spacing on rooftops (m)
MAX_RAY_LEN_M   = 50.0      # ray length (≤ buffer radius)

# Sun sampling (Mar 21; azimuths from pvlib; elevations from your lists)
LAT, LON, ALT_M = 23.8103, 90.4125, 10
TIMEZONE        = "Asia/Dhaka"
YEAR, MONTH, DAY= 2025, 3, 21
HOURS_AM        = [8,9,10,11]            # 4 stamps; we’ll expand to 5 slots to match your 5 elevations
HOURS_PM        = [13,14,15,16]          # 4 stamps

# Elevation lists from your screenshots (deg)
ELEV_AM_DEG = [15.82, 26.34, 35.02, 40.86, 42.78]   # 5 values
ELEV_PM_DEG = [40.32, 34.05, 25.08, 14.39]          # 4 values

# Sector filters (bearing target->neighbor, deg from north CW)
SECTOR_AM = (90.0, 180.0)    # [90,180)
SECTOR_PM = (180.0, 270.0)   # [180,270)

os.makedirs(STRUCT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def run_cmd(cmd, check=True):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p

def ensure_tool(name):
    probe = "ogr2ogr" if name == "gdal-bin" else name
    p = subprocess.run(["bash","-lc", f"command -v {probe} || true"], stdout=subprocess.PIPE, text=True)
    if not p.stdout.strip():
        run_cmd(["apt-get","update"])
        run_cmd(["apt-get","-y","install", name])

def unzip_to_dir(zip_path, out_dir):
    ensure_tool("unzip")
    run_cmd(["unzip","-o", zip_path, "-d", out_dir])
    shp_files = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(".shp"):
                shp_files.append(os.path.join(root, f))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found after unzipping {zip_path}")
    shp_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return shp_files[0]

def to_gpkg_if_needed(shp_path):
    ensure_tool("gdal-bin")
    gpkg_path = shp_path.rsplit(".", 1)[0] + ".gpkg"
    if not os.path.exists(gpkg_path):
        run_cmd([
            "ogr2ogr","-skipfailures","-f","GPKG", gpkg_path, shp_path,
            "-lco","ENCODING=UTF-8","-nlt","PROMOTE_TO_MULTI"
        ])
    return gpkg_path

def safe_read(path, use_cols=None):
    try:
        df = gpd.read_file(path)
    except Exception:
        try:
            cpg = os.path.splitext(path)[0] + ".cpg"
            enc = "cp1252"
            if os.path.exists(cpg):
                with open(cpg, "r", encoding="utf-8", errors="ignore") as f:
                    enc = f.read().strip() or enc
            df = gpd.read_file(path, engine="fiona", encoding=enc)
        except Exception:
            gpkg = to_gpkg_if_needed(path)
            df = gpd.read_file(gpkg)
    if use_cols:
        keep = [c for c in use_cols if c in df.columns]
        if keep:
            df = df[keep + ["geometry"]]
    return df

def force_2d(geom):
    try:
        return shp_transform(lambda x,y,z=None:(x,y), geom)
    except Exception:
        return geom

def ensure_projected(gdf, epsg):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(force_2d)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    if (gdf.crs.to_epsg() or gdf.crs.to_string()) != f"EPSG:{epsg}":
        gdf = gdf.to_crs(epsg=epsg)
    return gdf

def height_from_floors(series, floor_h=FLOOR_HEIGHT_M):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return s * float(floor_h)

def bearing_deg(p_from: Point, p_to: Point) -> float:
    dx = p_to.x - p_from.x
    dy = p_to.y - p_from.y
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0

def edge_distance(poly_a, poly_b) -> float:
    a, b = nearest_points(poly_a, poly_b)
    return a.distance(b)

def try_write_shp_then_gpkg(gdf, shp_path, gpkg_path, layer=None):
    try:
        gdf.to_file(shp_path)
        print(f"Saved SHP: {shp_path}")
    except Exception as e:
        print(f"SHP write failed ({e}); writing GPKG instead.")
        if layer is None:
            layer = os.path.splitext(os.path.basename(gpkg_path))[0]
        gdf.to_file(gpkg_path, layer=layer, driver="GPKG")
        print(f"Saved GPKG: {gpkg_path}")

def grid_points_in_polygon(poly, spacing=2.0):
    """Return list of Points on a regular grid clipped to polygon."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + spacing/2.0, maxx, spacing)
    ys = np.arange(miny + spacing/2.0, maxy, spacing)
    pts = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if poly.contains(p):
                pts.append(p)
    return pts

def ray_endpoint(point: Point, az_deg: float, length_m: float):
    """0°=north, clockwise positive."""
    dx = math.sin(math.radians(az_deg))
    dy = math.cos(math.radians(az_deg))
    return Point(point.x + dx*length_m, point.y + dy*length_m)

def first_hit_on_ray(point: Point, az_deg: float, max_len: float, gdf_cand: gpd.GeoDataFrame):
    """
    Cast a line from point toward az_deg for max_len.
    Return (d_hit, height_of_building) for the NEAREST building hit; None if no hit.
    """
    if gdf_cand.empty:
        return None
    end = ray_endpoint(point, az_deg, max_len)
    ray = LineString([point, end])

    # quick cull by bbox
    sidx = gdf_cand.sindex
    idxs = list(sidx.intersection(ray.bounds))
    if not idxs:
        return None
    sub = gdf_cand.iloc[idxs]

    best_d = None
    best_h = None
    for _, row in sub.iterrows():
        geom = row.geometry
        inter = ray.intersection(geom.boundary)
        if inter.is_empty:
            continue
        # normalize to list of points
        pts = []
        gtype = inter.geom_type
        if gtype == "Point":
            pts = [inter]
        elif gtype in ("MultiPoint",):
            pts = list(inter.geoms)
        elif gtype in ("LineString","MultiLineString"):
            # touch along edge → use endpoints as entry/exit
            if gtype == "LineString":
                coords = list(inter.coords)
                pts = [Point(coords[0]), Point(coords[-1])]
            else:
                for ln in inter.geoms:
                    c = list(ln.coords)
                    pts.extend([Point(c[0]), Point(c[-1])])
        else:
            # fallback: closest point on ray to polygon
            pts = [nearest_points(ray, geom)[0]]

        for pt in pts:
            d = pt.distance(point)
            if d <= 1e-6 or d > max_len:
                continue
            if (best_d is None) or (d < best_d):
                best_d = d
                best_h = float(row["height_m"])
    if best_d is None:
        return None
    return best_d, best_h

def pvlib_azimuths_for_hours(year, month, day, hours):
    idx = [pd.Timestamp(year=year, month=month, day=day, hour=h, tz=TIMEZONE) for h in hours]
    sol = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(idx), LAT, LON, altitude=ALT_M)
    return list(sol["azimuth"].values)

# =========================
# LOAD + PREP
# =========================
struct_shp = unzip_to_dir(STRUCT_ZIP, STRUCT_DIR)
target_shp = unzip_to_dir(TARGET_ZIP, TARGET_DIR)

# Keep Floor + Str_Use1 if present
gdf_structs = safe_read(struct_shp, use_cols=[HEIGHT_FIELD, KEEP_USE_FIELD])
gdf_targets = safe_read(target_shp, use_cols=[HEIGHT_FIELD, KEEP_USE_FIELD])

gdf_structs = ensure_projected(gdf_structs, EPSG_METERS)
gdf_targets = ensure_projected(gdf_targets, EPSG_METERS)

# Heights (meters)
gdf_structs["height_m"] = height_from_floors(gdf_structs.get(HEIGHT_FIELD, 0))
gdf_targets["height_m"] = height_from_floors(gdf_targets.get(HEIGHT_FIELD, 0))

# Ensure Str_Use1 exists
for df in (gdf_structs, gdf_targets):
    if KEEP_USE_FIELD not in df.columns:
        df[KEEP_USE_FIELD] = ""

# Centroids & sindex
gdf_structs["cen"] = gdf_structs.geometry.centroid
gdf_targets["cen"] = gdf_targets.geometry.centroid
sindex_structs     = gdf_structs.sindex

# Sun azimuths (pvlib) for Mar 21
SUN_AZ_AM = pvlib_azimuths_for_hours(YEAR, MONTH, DAY, HOURS_AM)
# expand to 5 to match ELEV_AM_DEG
if len(SUN_AZ_AM) < len(ELEV_AM_DEG):
    SUN_AZ_AM = SUN_AZ_AM + [SUN_AZ_AM[-1]]*(len(ELEV_AM_DEG)-len(SUN_AZ_AM))
SUN_AZ_PM = pvlib_azimuths_for_hours(YEAR, MONTH, DAY, HOURS_PM)

# =========================
# MAIN: per-target, grid + ray
# =========================
records = []
for tidx, tgt in tqdm(gdf_targets.iterrows(), total=len(gdf_targets), desc="Grid+Ray shading"):
    t_geom = tgt.geometry
    t_cen  = tgt["cen"]
    h_tgt  = float(tgt["height_m"])

    # Skip degenerate or tiny roofs
    if t_geom.is_empty or t_geom.area < (GRID_SPACING_M**2)/4:
        records.append((tidx, [0]*5, [0]*4, 0, 0, 0)); continue

    # 50 m buffer and taller neighbors
    buf = t_geom.buffer(BUFFER_RADIUS_M)
    cand_idx = list(sindex_structs.intersection(buf.bounds))
    if not cand_idx:
        records.append((tidx, [0]*5, [0]*4, 0, 0, 0)); continue

    cand = gdf_structs.iloc[cand_idx].copy()
    cand = cand[cand.intersects(buf)]
    cand = cand[cand["height_m"] > h_tgt]
    if cand.empty:
        records.append((tidx, [0]*5, [0]*4, 0, 0, 0)); continue

    # Bearings & sector split
    cand["bearing"] = cand["cen"].apply(lambda p: bearing_deg(t_cen, p))
    cand_am = cand[(cand["bearing"] >= SECTOR_AM[0]) & (cand["bearing"] < SECTOR_AM[1])].copy()
    cand_pm = cand[(cand["bearing"] >= SECTOR_PM[0]) & (cand["bearing"] < SECTOR_PM[1])].copy()

    # Grid over the target roof
    pts = grid_points_in_polygon(t_geom, spacing=GRID_SPACING_M)
    if not pts:
        records.append((tidx, [0]*5, [0]*4, 0, 0, 0)); continue

    # --- AM steps (5) ---
    if cand_am.empty:
        pct_am = [0.0]*len(ELEV_AM_DEG)
    else:
        pct_am = []
        for k, elev in enumerate(ELEV_AM_DEG):
            az = SUN_AZ_AM[k] if k < len(SUN_AZ_AM) else SUN_AZ_AM[-1]
            tan_e = math.tan(math.radians(elev))
            shaded_count = 0
            for p in pts:
                hit = first_hit_on_ray(p, az, MAX_RAY_LEN_M, cand_am)
                if not hit:
                    continue
                d_hit, h_nb = hit
                Hshadow = h_nb - d_hit * tan_e
                if Hshadow >= h_tgt and Hshadow >= 0:
                    shaded_count += 1
            pct_am.append(100.0 * shaded_count / len(pts))

    # --- PM steps (4) ---
    if cand_pm.empty:
        pct_pm = [0.0]*len(ELEV_PM_DEG)
    else:
        pct_pm = []
        for k, elev in enumerate(ELEV_PM_DEG):
            az = SUN_AZ_PM[k] if k < len(SUN_AZ_PM) else SUN_AZ_PM[-1]
            tan_e = math.tan(math.radians(elev))
            shaded_count = 0
            for p in pts:
                hit = first_hit_on_ray(p, az, MAX_RAY_LEN_M, cand_pm)
                if not hit:
                    continue
                d_hit, h_nb = hit
                Hshadow = h_nb - d_hit * tan_e
                if Hshadow >= h_tgt and Hshadow >= 0:
                    shaded_count += 1
            pct_pm.append(100.0 * shaded_count / len(pts))

    am_mean = float(np.mean(pct_am)) if pct_am else 0.0
    pm_mean = float(np.mean(pct_pm)) if pct_pm else 0.0
    day_max = float(max(pct_am + pct_pm)) if (pct_am or pct_pm) else 0.0

    records.append((tidx, pct_am, pct_pm, am_mean, pm_mean, day_max))

# =========================
# ATTACH METRICS + AREA + SAVE
# =========================
def pad(lst, n):
    return (lst + [0.0]*n)[:n]

rows = []
for ridx, am_list, pm_list, am_mean, pm_mean, day_max in records:
    am5 = pad(am_list, 5)
    pm4 = pad(pm_list, 4)
    row = {
        "_row": ridx,
        "Pct_AM1": am5[0], "Pct_AM2": am5[1], "Pct_AM3": am5[2], "Pct_AM4": am5[3], "Pct_AM5": am5[4],
        "Pct_PM1": pm4[0], "Pct_PM2": pm4[1], "Pct_PM3": pm4[2], "Pct_PM4": pm4[3],
        "Pct_AM_Mn": am_mean, "Pct_PM_Mn": pm_mean, "Pct_Day_Mx": day_max,
    }
    rows.append(row)

res = pd.DataFrame(rows).set_index("_row")
gdf_targets = gdf_targets.join(res)

# Rooftop area (m²) in projected CRS
gdf_targets["Shape_Area"] = gdf_targets.geometry.area

# Choose a single "% shaded area" summary: daily maximum fraction
gdf_targets["PctAreaMx"] = gdf_targets["Pct_Day_Mx"]
gdf_targets["AreaShdMx"] = gdf_targets["Shape_Area"] * gdf_targets["PctAreaMx"] / 100.0

# Flag shaded/unshaded using any % > 0
gdf_targets["Shaded"] = (gdf_targets[["Pct_AM1","Pct_AM2","Pct_AM3","Pct_AM4","Pct_AM5",
                                      "Pct_PM1","Pct_PM2","Pct_PM3","Pct_PM4"]].max(axis=1) > 0.0)

# Keep fields (short names are Shapefile-safe)
keep_cols = []
if KEEP_USE_FIELD in gdf_targets.columns:
    keep_cols.append(KEEP_USE_FIELD)
if HEIGHT_FIELD in gdf_targets.columns:
    keep_cols.append(HEIGHT_FIELD)
keep_cols += ["height_m","Shape_Area",
              "Pct_AM1","Pct_AM2","Pct_AM3","Pct_AM4","Pct_AM5",
              "Pct_PM1","Pct_PM2","Pct_PM3","Pct_PM4",
              "Pct_AM_Mn","Pct_PM_Mn","Pct_Day_Mx",
              "PctAreaMx","AreaShdMx",
              "Shaded","geometry"]
gdf_targets = gdf_targets[keep_cols]

gdf_shaded   = gdf_targets[gdf_targets["Shaded"]].copy()
gdf_unshaded = gdf_targets[~gdf_targets["Shaded"]].copy()

# Write files (SHP → fallback to GPKG)
shaded_shp    = os.path.join(OUT_DIR, "Targets_Shaded_Pct_50m.shp")
unshaded_shp  = os.path.join(OUT_DIR, "Targets_Unshaded_Pct_50m.shp")
shaded_gpkg   = os.path.join(OUT_DIR, "Targets_Shaded_Pct_50m.gpkg")
unshaded_gpkg = os.path.join(OUT_DIR, "Targets_Unshaded_Pct_50m.gpkg")

try_write_shp_then_gpkg(gdf_shaded,   shaded_shp,   shaded_gpkg,   layer="shaded")
try_write_shp_then_gpkg(gdf_unshaded, unshaded_shp, unshaded_gpkg, layer="unshaded")

# Optional zip
zip_path = shutil.make_archive(ZIP_BASE, "zip", OUT_DIR)
tds = time.time() - t0
print("✅ Done.")
print("Folder:", OUT_DIR)
print("ZIP:", zip_path)
print(f"Total runtime: {tds:.1f} s (~{tds/60:.2f} min)")
