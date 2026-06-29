"""
Build a shaded-relief terrain overlay PNG from a Vintage Story save's mapchunk
table (the explored area). Elevation + hillshade + water line -> a recognisable
map backdrop for the ore dots.

Returns bounds so the app can place/scale the image in world coordinates.
"""
from __future__ import annotations
import sqlite3
import numpy as np
from PIL import Image

CHUNK = 32


def _decode_pos(pos: int):
    return pos & 0x1FFFFF, (pos >> 27) & 0x1FFFFF      # chunkX, chunkZ


def _rv(b, i):
    shift = 0
    v = 0
    while True:
        x = b[i]; i += 1
        v |= (x & 0x7F) << shift
        if not (x & 0x80):
            break
        shift += 7
    return v, i


def _heights(blob: bytes):
    """field 3 = terrain heightmap, 1024 varints (z*32+x). It is NOT always the
    first field (some chunks lead with mod-data), so walk the protobuf to find
    it wherever it sits."""
    i = 0
    n = len(blob)
    out = []
    while i < n:
        tag, i = _rv(blob, i)
        f, wt = tag >> 3, tag & 7
        if wt == 0:                       # varint
            v, i = _rv(blob, i)
            if f == 3:
                out.append(v)
                if len(out) >= CHUNK * CHUNK:
                    break
        elif wt == 2:                     # length-delimited
            ln, i = _rv(blob, i)
            if f == 3:                     # packed varints
                end = i + ln
                while i < end:
                    v, i = _rv(blob, i)
                    out.append(v)
            else:
                i += ln
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    return out if len(out) == CHUNK * CHUNK else None


# elevation -> colour ramp (deep water .. snow)
_STOPS = np.array([95, 109, 110, 113, 122, 133, 145, 165], dtype=np.float32)
_COLS = np.array([
    (35, 60, 110), (70, 110, 165), (212, 200, 150), (110, 160, 92),
    (74, 120, 70), (128, 108, 84), (172, 156, 128), (236, 236, 242),
], dtype=np.float32)


def build_overlay(con: sqlite3.Connection, out_path: str, sealevel: int = 110) -> dict | None:
    rows = con.execute("SELECT position, data FROM mapchunk").fetchall()
    if not rows:
        return None
    cells = {}
    minx = minz = 1 << 30
    maxx = maxz = -(1 << 30)
    for pos, data in rows:
        h = _heights(bytes(data))
        if h is None:
            continue
        cx, cz = _decode_pos(pos)
        cells[(cx, cz)] = np.asarray(h, np.float32).reshape(CHUNK, CHUNK)  # [z,x]
        minx = min(minx, cx); maxx = max(maxx, cx)
        minz = min(minz, cz); maxz = max(maxz, cz)
    if not cells:
        return None

    cw = (maxx - minx + 1) * CHUNK
    ch = (maxz - minz + 1) * CHUNK
    H = np.full((ch, cw), np.nan, np.float32)
    for (cx, cz), grid in cells.items():
        r = (cz - minz) * CHUNK
        c = (cx - minx) * CHUNK
        H[r:r + CHUNK, c:c + CHUNK] = grid

    explored = ~np.isnan(H)
    fill = float(np.nanmedian(H))
    Hc = np.where(explored, H, fill).astype(np.float32)

    # hillshade (light from NW, 45deg)
    gy, gx = np.gradient(Hc)
    slope = np.arctan(np.hypot(gx, gy) * 2.0)      # exaggerate relief a bit
    aspect = np.arctan2(-gx, gy)
    alt, az = np.deg2rad(45), np.deg2rad(315)
    shade = (np.sin(alt) * np.cos(slope)
             + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    shade = np.clip(0.55 + 0.55 * shade, 0.35, 1.15)[..., None]

    # base colour from elevation ramp
    clip = np.clip(Hc, _STOPS[0], _STOPS[-1])
    rgb = np.empty(Hc.shape + (3,), np.float32)
    for ch_i in range(3):
        rgb[..., ch_i] = np.interp(clip, _STOPS, _COLS[:, ch_i])
    # tiny extra darkening for deep water already handled by ramp
    rgb = np.clip(rgb * shade, 0, 255)

    alpha = np.where(explored, 235, 0).astype(np.uint8)
    out = np.dstack([rgb.astype(np.uint8), alpha])
    im = Image.fromarray(out, "RGBA")
    # cap the texture size: it's a background layer (the app scales it to world
    # size anyway), and a smaller texture pans/zooms far more smoothly
    MAXDIM = 2800
    f = min(1.0, MAXDIM / max(cw, ch))
    if f < 1.0:
        im = im.resize((max(1, int(cw * f)), max(1, int(ch * f))), Image.BILINEAR)
    im.save(out_path, optimize=True)

    return {
        "png": out_path,
        "x0": minx * CHUNK,            # world coord of image top-left (X)
        "z0": minz * CHUNK,            # world coord of image top-left (Z)
        "w": cw, "h": ch,             # image size in blocks (= pixels)
    }


# colour ramps (intensity 0..1 -> RGB): trees = green, rocks = purple/magenta
_TD_STOPS = np.array([0.0, 0.25, 0.55, 1.0])
_TREE_RAMP = (_TD_STOPS, np.array([40, 70, 150, 225]),
              np.array([120, 200, 240, 255]), np.array([45, 60, 80, 170]))
_ROCK_RAMP = (_TD_STOPS, np.array([95, 150, 210, 250]),
              np.array([55, 60, 70, 180]), np.array([165, 205, 215, 245]))
# sediment ramps: sand = amber/yellow, gravel = slate grey
_SAND_RAMP = (_TD_STOPS, np.array([200, 235, 250, 255]),
              np.array([140, 185, 215, 245]), np.array([40, 60, 90, 170]))
_GRAVEL_RAMP = (_TD_STOPS, np.array([90, 140, 190, 235]),
                np.array([110, 160, 195, 235]), np.array([135, 185, 210, 245]))
# clay ramps: colour matches the chosen clay (blue / red / all = purple)
_CLAY_BLUE_RAMP = (_TD_STOPS, np.array([40, 60, 95, 150]),
                   np.array([90, 130, 175, 215]), np.array([200, 230, 245, 255]))
_CLAY_RED_RAMP = (_TD_STOPS, np.array([200, 235, 250, 255]),
                  np.array([70, 100, 140, 195]), np.array([55, 65, 85, 150]))
_CLAY_ALL_RAMP = (_TD_STOPS, np.array([120, 165, 205, 240]),
                  np.array([55, 75, 110, 170]), np.array([160, 200, 225, 250]))


def _density_heatmap(cache, data_key, type_idx, ramp, out_path):
    """Render a per-chunk count heatmap (data_key e.g. 'trees'/'rocks') for one
    type (or all if type_idx is None), composited onto the terrain -> one image."""
    from PIL import ImageFilter
    data = cache.get(data_key)
    ov = cache.get("overlay")
    if not data or not ov:
        return None
    x0, z0, w, h = ov["x0"], ov["z0"], ov["w"], ov["h"]
    cols = max(1, w // CHUNK); rows = max(1, h // CHUNK)
    min_cx, min_cz = x0 // CHUNK, z0 // CHUNK
    dens = np.zeros((rows, cols), float)
    for pos, counts in data.items():
        c = int(counts.sum()) if type_idx is None else int(counts[type_idx])
        if not c:
            continue
        cx, cz = _decode_pos(pos)
        r, cc = cz - min_cz, cx - min_cx
        if 0 <= r < rows and 0 <= cc < cols:
            dens[r, cc] += c
    if not dens.any():
        return None
    n = np.log1p(dens)
    n /= (n.max() or 1.0)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8), "L")
                   .filter(ImageFilter.GaussianBlur(0.8)), float) / 255.0
    stops, R, G, B = ramp
    r = np.interp(n, stops, R)
    g = np.interp(n, stops, G)
    b = np.interp(n, stops, B)
    a = np.clip(n ** 0.6 * 220, 0, 220)
    a[n < 0.03] = 0
    heat = Image.fromarray(np.dstack([r, g, b, a]).astype(np.uint8), "RGBA")
    base = None
    try:
        if ov.get("png"):
            base = Image.open(ov["png"]).convert("RGBA")
    except Exception:
        base = None
    if base is not None:
        out = Image.alpha_composite(base, heat.resize(base.size, Image.BILINEAR))
    else:
        target = int(min(1600, max(cols, rows) * 5))
        f = target / max(cols, rows)
        out = heat.resize((max(1, int(cols * f)), max(1, int(rows * f))), Image.BILINEAR)
    out.save(out_path)
    return {"x0": x0, "z0": z0, "w": w, "h": h}


def build_tree_density(cache, species_idx, out_path):
    return _density_heatmap(cache, "trees", species_idx, _TREE_RAMP, out_path)


def build_rock_density(cache, rock_idx, out_path):
    return _density_heatmap(cache, "rocks", rock_idx, _ROCK_RAMP, out_path)


def build_gravel_density(cache, type_idx, out_path):
    return _density_heatmap(cache, "gravel", type_idx, _GRAVEL_RAMP, out_path)


def build_sand_density(cache, type_idx, out_path):
    return _density_heatmap(cache, "sand", type_idx, _SAND_RAMP, out_path)


def build_clay_density(cache, type_idx, out_path):
    # type_idx: None = all clay (purple), 0 = blue clay, 1 = red clay
    ramp = (_CLAY_BLUE_RAMP if type_idx == 0 else
            _CLAY_RED_RAMP if type_idx == 1 else _CLAY_ALL_RAMP)
    return _density_heatmap(cache, "clay", type_idx, ramp, out_path)


if __name__ == "__main__":
    import sys, time
    if len(sys.argv) > 1:
        save = sys.argv[1]
    else:
        from scanner import find_saves
        saves = find_saves()
        if not saves:
            sys.exit("usage: python map_overlay.py <save.vcdbs> [out.png]")
        save = saves[0][1]
    out = sys.argv[2] if len(sys.argv) > 2 else "test_overlay.png"
    con = sqlite3.connect(f"file:{save}?mode=ro", uri=True)
    t = time.time()
    info = build_overlay(con, out)
    print(f"built in {time.time()-t:.1f}s -> {info}")
