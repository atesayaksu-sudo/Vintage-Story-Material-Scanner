"""
Vintage Story ore scanner.

Reads a .vcdbs save file and extracts the exact world coordinates of every ore
block, using Vintage Story's *own* chunk decoder (loaded from the installed game
DLLs via pythonnet). No reverse-engineering, no heatmaps - real block positions.

Results are cached to disk per save:
  * the app loads the last scan instantly (no re-scan needed),
  * a "rescan" only decodes chunks whose data actually changed (incremental).

Public API:
    find_game_dir()                  -> game install path or None
    find_saves()                     -> [(name, path, size), ...]
    Scanner(game_dir)                -> .scan_to_cache(...) / re-decode
    load_cache(save_path)            -> cache dict or None
    save_cache(cache)
    cluster_from_cache(cache, size)  -> [OreCluster, ...]
"""

from __future__ import annotations
import os
import zlib
import time
import pickle
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Locating the game install and saves
# ---------------------------------------------------------------------------

_APPDATA = os.environ.get("APPDATA", os.path.expanduser(r"~\AppData\Roaming"))
_LOCAL = os.environ.get("LOCALAPPDATA", os.path.expanduser(r"~\AppData\Local"))
_PROGFILES = os.environ.get("ProgramFiles", r"C:\Program Files")
_PROGFILES86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

# the game lets users relocate its data folder via this env var; fall back to default
_DATA_DIR = (os.environ.get("VINTAGE_STORY_DATA_PATH")
             or os.path.join(_APPDATA, "VintagestoryData"))

GAME_DIR_CANDIDATES = [
    os.environ.get("VINTAGE_STORY"),                 # set by the official installer
    os.path.join(_APPDATA, "Vintagestory"),
    os.path.join(_PROGFILES, "Vintagestory"),
    os.path.join(_PROGFILES86, "Vintagestory"),
    os.path.join(_LOCAL, "Vintagestory"),
    os.path.join(_PROGFILES86, "Steam", "steamapps", "common", "Vintagestory"),
    os.path.join(_PROGFILES, "Steam", "steamapps", "common", "Vintagestory"),
]
SAVES_DIR = os.path.join(_DATA_DIR, "Saves")
# all writable app data lives here (works when frozen into an .exe, where the
# app's own folder is read-only / a temp extraction)
APP_DIR = os.path.join(_LOCAL, "VSOreFinder")
CACHE_DIR = os.path.join(APP_DIR, "cache")
ASSETS_DIR = os.path.join(APP_DIR, "assets")   # generated PNGs flet serves
CACHE_VERSION = 13

# tree species detected from `log-grown-<species>-*` blocks (for tree-density map)
TREE_SPECIES = ["oak", "birch", "maple", "pine", "acacia", "kapok", "larch",
                "redwood", "walnut", "ebony", "purpleheart", "baldcypress"]

# in-ground rock types detected from `rock-<type>` blocks (for rock-layer map)
ROCK_TYPES = ["granite", "basalt", "andesite", "limestone", "chalk", "sandstone",
              "shale", "claystone", "conglomerate", "slate", "phyllite", "schist",
              "chert", "peridotite", "kimberlite", "suevite", "scoria", "tuff",
              "obsidian", "bauxite", "halite", "travertine", "whitemarble",
              "redmarble", "greenmarble"]

# raw clay deposits for the clay heatmap (rawclay-blue-* / rawclay-red-*);
# fireclay is handled separately as a tracked material
CLAY_TYPES = ["blue", "red"]


def find_game_dir() -> str | None:
    for c in GAME_DIR_CANDIDATES:
        if c and os.path.exists(os.path.join(c, "VintagestoryLib.dll")):
            return c
    return None


def find_saves(saves_dir: str = SAVES_DIR) -> list[tuple[str, str, int]]:
    out = []
    if os.path.isdir(saves_dir):
        for f in os.listdir(saves_dir):
            if f.lower().endswith(".vcdbs"):
                p = os.path.join(saves_dir, f)
                out.append((os.path.splitext(f)[0], p, os.path.getsize(p)))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def cache_path(save_path: str) -> str:
    key = "%08x" % (zlib.crc32(save_path.encode("utf8")) & 0xFFFFFFFF)
    base = os.path.splitext(os.path.basename(save_path))[0]
    return os.path.join(CACHE_DIR, f"{base}.{key}.orecache")


def _player_pos_from_playerdata(con, mapx: int = 1024000, mapz: int = 1024000):
    """The player's last-saved world (X, Z). It's stored as repeated coordinate
    doubles in the playerdata blob; the real pair is the one that recurs. Used
    for calibration: origin = this position - the in-game coords the player
    reads at that same spot. Pure Python — no game runtime needed."""
    import struct
    from collections import Counter
    lo = 1000.0
    cnt: Counter = Counter()
    try:
        rows = con.execute("SELECT * FROM playerdata").fetchall()
    except Exception:
        return None
    for r in rows:
        blob = next((bytes(x) for x in r if isinstance(x, (bytes, bytearray))), None)
        if not blob:
            continue
        ds = []
        for i in range(len(blob) - 8):
            d = struct.unpack_from("<d", blob, i)[0]
            if d == d and lo < d < max(mapx, mapz):     # finite, coordinate-sized
                ds.append((i, d))
        for k, (oi, x) in enumerate(ds):
            if not (lo < x < mapx):
                continue
            for oj, z in ds[k + 1:]:
                if oj - oi > 24:        # X and Z sit next to each other
                    break
                if lo < z < mapz:
                    # floor to match the in-game HUD's block coordinates
                    cnt[(int(x), int(z))] += 1
                    break
    if not cnt:
        return None
    (sx, sz), _ = cnt.most_common(1)[0]
    return int(sx), int(sz)


def player_pos_from_save(save_path: str):
    """Open a save read-only and return the player's last-saved (X, Z)."""
    try:
        con = sqlite3.connect(f"file:{save_path}?mode=ro", uri=True)
        try:
            return _player_pos_from_playerdata(con)
        finally:
            con.close()
    except Exception:
        return None


SETTINGS_PATH = os.path.join(CACHE_DIR, "settings.json")


def load_settings() -> dict:
    """User overrides that survive restarts: a custom game folder and any save
    files browsed-to from outside the default Saves directory."""
    try:
        import json
        with open(SETTINGS_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(data: dict):
    try:
        import json
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def is_game_dir(path: str | None) -> bool:
    return bool(path and os.path.exists(os.path.join(path, "VintagestoryLib.dll")))


# ---------------------------------------------------------------------------
# Ore code parsing
# ---------------------------------------------------------------------------

GRADES = {"poor", "medium", "rich", "bountiful", "low", "high"}

MINERAL_INFO = {
    "nativecopper": ("Native copper", "copper"), "malachite": ("Malachite", "copper"),
    "cuprite": ("Cuprite", "copper"), "tetrahedrite": ("Tetrahedrite", "copper"),
    "cassiterite": ("Cassiterite", "tin"), "sphalerite": ("Sphalerite", "zinc"),
    "bismuthinite": ("Bismuthinite", "bismuth"), "magnetite": ("Magnetite", "iron"),
    "limonite": ("Limonite", "iron"), "hematite": ("Hematite", "iron"),
    "goethite": ("Goethite", "iron"), "siderite": ("Siderite", "iron"),
    "nativegold": ("Native gold", "gold"), "nativesilver": ("Native silver", "silver"),
    "galena": ("Galena", "lead/silver"), "pentlandite": ("Pentlandite", "nickel"),
    "chromite": ("Chromite", "chromium"), "ilmenite": ("Ilmenite", "titanium"),
    "rhodochrosite": ("Rhodochrosite", "manganese"), "quartz": ("Quartz", "quartz"),
    "sulfur": ("Sulfur", "sulfur"), "graphite": ("Graphite", "graphite"),
    "lapislazuli": ("Lapis lazuli", "lapis"), "diamond": ("Diamond", "diamond"),
    "emerald": ("Emerald", "emerald"), "olivine_peridot": ("Peridot", "peridot"),
    "corundum_ruby": ("Ruby", "ruby"), "corundum_sapphire": ("Sapphire", "sapphire"),
    "borax": ("Borax", "borax"), "fluorite": ("Fluorite", "fluorite"),
    "phosphorite": ("Phosphorite", "phosphorus"), "alum": ("Alum", "alum"),
    "saltpeter": ("Saltpeter", "saltpeter"), "halite": ("Halite", "salt"),
    "sylvite": ("Sylvite", "potash"),
    "kernite": ("Kernite", "borax"), "cinnabar": ("Cinnabar", "mercury"),
    "uranium": ("Uraninite", "uranium"),
    "lignite": ("Lignite", "lignite"), "bituminouscoal": ("Bituminous coal", "bituminous"),
    "anthracite": ("Anthracite", "anthracite"),
}
METAL_PRIORITY = {
    "gold": 100, "silver": 90, "lead/silver": 85, "meteoriciron": 82,
    "iron": 80, "copper": 70,
    "tin": 65, "zinc": 60, "nickel": 55, "titanium": 50, "chromium": 45,
    "bismuth": 40, "diamond": 95, "emerald": 88, "ruby": 80, "sapphire": 78,
    "peridot": 30, "manganese": 35, "mercury": 42, "uranium": 48,
    # non-metal useful minerals
    "borax": 25, "sulfur": 24, "saltpeter": 26, "alum": 20, "fluorite": 22,
    "phosphorus": 21, "potash": 23, "graphite": 28, "salt": 19, "lapis": 60,
    "fireclay": 27, "olivine": 26,
    # coal
    "anthracite": 18, "bituminous": 16, "lignite": 14,
    # misc
    "beehive": 30, "resin": 29,
}
GRADE_RANK = {"": 0, "poor": 1, "low": 1, "medium": 2, "rich": 3, "high": 3, "bountiful": 4}


@dataclass
class OreType:
    code: str
    grade: str
    mineral: str
    rock: str
    label: str
    metal: str


def parse_ore_code(code: str) -> OreType:
    toks = code.split("-")
    grade = ""
    i = 1
    if len(toks) > 1 and toks[1] in GRADES:
        grade = toks[1]
        i = 2
    mineral = toks[i] if i < len(toks) else "?"
    rock = "-".join(toks[i + 1:]) if i + 1 < len(toks) else ""
    label, metal = MINERAL_INFO.get(mineral, (None, None))
    if label is None and "_" in mineral:
        second = mineral.split("_", 1)[1]
        label, metal = MINERAL_INFO.get(second, (None, None))
    if label is None:
        label = mineral.replace("_", " ").title()
        metal = mineral
    return OreType(code, grade, mineral, rock, label, metal)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class OreCluster:
    metal: str
    mineral: str
    best_grade: str
    count: int
    cx: int
    cy: int
    cz: int
    ymin: int
    ymax: int
    samples: list = field(default_factory=list)

    @property
    def priority(self) -> int:
        return METAL_PRIORITY.get(self.metal, 10) * 10 + GRADE_RANK.get(self.best_grade, 0)


def _varint(b: bytes, i: int) -> tuple[int, int]:
    shift = 0
    val = 0
    while True:
        x = b[i]
        i += 1
        val |= (x & 0x7F) << shift
        if not (x & 0x80):
            break
        shift += 7
    return val, i


def _decode_pos(pos: int) -> tuple[int, int, int]:
    cx = pos & 0x1FFFFF
    cz = (pos >> 27) & 0x1FFFFF
    cy = (pos >> 54) & 0x1FF
    return cx, cy, cz


# ---------------------------------------------------------------------------
# Clustering (shared by scan + on-the-fly merge-radius changes)
# ---------------------------------------------------------------------------

def cluster_from_cache(cache: dict, cluster_size: int = 12,
                       metals: Iterable[str] | None = None) -> list[OreCluster]:
    ore = cache["ore"]                  # pos -> (idxs uint16, bids uint16)
    ore_types = cache["ore_types"]      # bid -> (metal, label, grade)
    wanted = set(metals) if metals else None
    cs = max(1, int(cluster_size))
    buckets: dict[tuple, dict] = {}

    for pos, (idxs, bids) in ore.items():
        cx, cy, cz = _decode_pos(pos)
        bx, by, bz = cx * 32, cy * 32, cz * 32
        ii = idxs.astype(np.int64)
        xs = bx + (ii & 31)
        zs = bz + ((ii >> 5) & 31)
        ys = by + ((ii >> 10) & 31)
        for k in range(len(idxs)):
            bid = int(bids[k])
            metal, label, grade = ore_types[bid]
            if wanted is not None and metal not in wanted:
                continue
            x, y, z = int(xs[k]), int(ys[k]), int(zs[k])
            key = (metal, x // cs, y // cs, z // cs)
            b = buckets.get(key)
            if b is None:
                b = dict(metal=metal, label=label, sx=0, sy=0, sz=0, n=0,
                         ymin=y, ymax=y, grade=grade, samples=[])
                buckets[key] = b
            b["sx"] += x; b["sy"] += y; b["sz"] += z; b["n"] += 1
            if y < b["ymin"]: b["ymin"] = y
            if y > b["ymax"]: b["ymax"] = y
            if GRADE_RANK.get(grade, 0) > GRADE_RANK.get(b["grade"], 0):
                b["grade"] = grade
            if len(b["samples"]) < 5:
                b["samples"].append((x, y, z))

    clusters = []
    for b in buckets.values():
        n = b["n"]
        clusters.append(OreCluster(
            metal=b["metal"], mineral=b["label"], best_grade=b["grade"], count=n,
            cx=b["sx"] // n, cy=b["sy"] // n, cz=b["sz"] // n,
            ymin=b["ymin"], ymax=b["ymax"], samples=b["samples"]))
    clusters.sort(key=lambda c: c.priority, reverse=True)
    return clusters


def load_cache(save_path: str) -> dict | None:
    p = cache_path(save_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            cache = pickle.load(f)
        if cache.get("version") != CACHE_VERSION:
            return None
        return cache
    except Exception:
        return None


def save_cache(cache: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = cache["cache_file"]
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------

NBLOCKS = 32 * 32 * 32
_RUNTIME_READY = False

# materials cached & selectable in the UI (sparse, vein-like).  Deliberately
# excludes super-abundant rock-formers (quartz, coal) for performance; rocks
# like limestone/chalk are handled separately.
TRACKED_MATERIALS = [
    # metals
    "gold", "silver", "lead/silver", "meteoriciron", "iron", "copper", "tin",
    "zinc", "nickel",
    "titanium", "chromium", "bismuth", "manganese", "mercury", "uranium",
    # useful non-metal minerals
    "borax", "sulfur", "saltpeter", "alum", "fluorite", "phosphorus", "potash",
    "graphite", "salt", "fireclay", "olivine",
    # coal
    "anthracite", "bituminous", "lignite",
    # gems
    "diamond", "emerald", "ruby", "sapphire", "peridot", "lapis",
    # misc surface finds
    "beehive", "resin",
]
# back-compat alias
DEFAULT_METALS = TRACKED_MATERIALS


class Scanner:
    def __init__(self, game_dir: str | None = None):
        self.game_dir = game_dir or find_game_dir()
        if not self.game_dir:
            raise RuntimeError(
                "Vintage Story install not found. Looked in:\n  "
                + "\n  ".join(GAME_DIR_CANDIDATES))
        self._init_runtime()

    def _init_runtime(self):
        global _RUNTIME_READY
        game = self.game_dir
        lib = os.path.join(game, "Lib")
        if not _RUNTIME_READY:
            from clr_loader import get_coreclr
            from pythonnet import set_runtime
            set_runtime(get_coreclr(
                runtime_config=os.path.join(game, "Vintagestory.runtimeconfig.json")))
            _RUNTIME_READY = True

        import clr  # noqa: F401
        import System
        from System.Reflection import Assembly, BindingFlags
        from System import ResolveEventHandler, AppDomain
        search = [game, lib]

        def resolver(sender, args):
            name = args.Name.split(",")[0]
            for base in search:
                p = os.path.join(base, name + ".dll")
                if os.path.exists(p):
                    return Assembly.LoadFrom(p)
            return None

        AppDomain.CurrentDomain.add_AssemblyResolve(ResolveEventHandler(resolver))
        asm = Assembly.LoadFrom(os.path.join(game, "VintagestoryLib.dll"))

        def get_types(a):
            try:
                return list(a.GetTypes())
            except System.Reflection.ReflectionTypeLoadException as e:
                return [t for t in e.Types if t is not None]

        types = {t.FullName: t for t in get_types(asm) if t.FullName}
        ALL = (BindingFlags.Public | BindingFlags.NonPublic
               | BindingFlags.Static | BindingFlags.Instance)
        from System import Array, Int32, Byte
        from System.Runtime.InteropServices import Marshal
        self._System = System
        self._UnpackBlocksTo = types["Vintagestory.Common.ChunkData"].GetMethod(
            "UnpackBlocksTo", ALL)
        self._Array, self._Int32, self._Byte, self._Marshal = Array, Int32, Byte, Marshal
        self._netout = Array.CreateInstance(Int32, NBLOCKS)
        self._npbuf = np.empty(NBLOCKS, dtype=np.int32)
        self._BF = ALL

        # protobuf-net + SaveGame type, for reading the world spawn point
        self._SaveGame = None
        self._pb_des = None
        try:
            pb = Assembly.LoadFrom(os.path.join(lib, "protobuf-net.dll"))
            self._SaveGame = next((t for t in types.values() if t.Name == "SaveGame"),
                                  None)
            ser = next((t for t in get_types(pb)
                        if t.FullName == "ProtoBuf.Serializer"), None)
            if ser:
                self._pb_des = next(
                    (m for m in ser.GetMethods(ALL)
                     if m.Name == "Deserialize" and not m.IsGenericMethod
                     and [p.ParameterType.Name for p in m.GetParameters()]
                     == ["Type", "Stream"]), None)
        except Exception:
            self._SaveGame = self._pb_des = None

    def _read_spawn(self, con) -> tuple[int, int] | None:
        """World spawn (= coordinate origin shown in-game). Tries the explicit
        DefaultSpawn, then the player's stored spawn, then the map middle."""
        if not (self._SaveGame and self._pb_des):
            return None
        try:
            row = con.execute("SELECT data FROM gamedata LIMIT 1").fetchone()
            from System.IO import MemoryStream
            ms = MemoryStream(self._Array[self._Byte](bytes(row[0])))
            sg = self._pb_des.Invoke(None, [self._SaveGame, ms])
            BF = self._BF

            def field(n):
                f = self._SaveGame.GetField(n, BF)
                return f.GetValue(sg) if f else None

            spawn = self._SaveGame.GetProperty("DefaultSpawn", BF)
            sp = spawn.GetValue(sg) if spawn else None
            if sp is not None:
                st = sp.GetType()
                x = st.GetField("X", BF).GetValue(sp)
                z = st.GetField("Z", BF).GetValue(sp)
                if x or z:
                    return int(x), int(z)
            # no explicit spawn: VS computes it at runtime and never stores it,
            # so we can't read it — fall back to the map middle. The user can
            # calibrate the exact origin from their in-game position in the app.
            msx, msz = field("MapSizeX"), field("MapSizeZ")
            if msx and msz:
                return int(msx) // 2, int(msz) // 2
        except Exception:
            pass
        return None

    def _decode(self, field1: bytes) -> np.ndarray:
        self._UnpackBlocksTo.Invoke(
            None, [self._netout, self._Array[self._Byte](field1), None, self._Int32(2)])
        self._Marshal.Copy(self._netout, 0,
                           self._System.IntPtr(self._npbuf.ctypes.data), NBLOCKS)
        return self._npbuf

    @staticmethod
    def _read_block_mapping(con) -> dict[int, str]:
        blob = con.execute("SELECT data FROM gamedata LIMIT 1").fetchone()[0]
        ix = blob.index(b"BlockIDs") + len(b"BlockIDs")
        assert blob[ix] == 0x12
        ix += 1
        mlen, ix = _varint(blob, ix)
        end = ix + mlen
        mapping: dict[int, str] = {}
        while ix < end:
            assert blob[ix] == 0x0A
            ix += 1
            el, ix = _varint(blob, ix)
            ee = ix + el
            bid = code = None
            j = ix
            while j < ee:
                tg = blob[j]
                j += 1
                if tg == 0x08:
                    bid, j = _varint(blob, j)
                elif tg == 0x12:
                    sl, j = _varint(blob, j)
                    code = blob[j:j + sl].decode("utf8", "replace")
                    j += sl
                else:
                    break
            if bid is not None:
                mapping[bid] = code
            ix = ee
        return mapping

    @staticmethod
    def _snapshot(save_path: str) -> tuple[str, str]:
        tmp = tempfile.mkdtemp(prefix="vsore_")
        base = os.path.join(tmp, "world.vcdbs")
        shutil.copy2(save_path, base)
        for ext in ("-wal", "-shm"):
            src = save_path + ext
            if os.path.exists(src):
                shutil.copy2(src, base + ext)
        return base, tmp

    def scan_to_cache(
        self,
        save_path: str,
        cluster_size: int = 12,
        incremental: bool = True,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Full or incremental scan; writes & returns the cache dict."""
        prev = load_cache(save_path) if incremental else None
        prev_hashes = prev["hashes"] if prev else {}
        prev_ore = prev["ore"] if prev else {}

        snap, tmpdir = self._snapshot(save_path)
        try:
            con = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
            mapping = self._read_block_mapping(con)
            tracked = set(TRACKED_MATERIALS)
            ore_parsed = {bid: parse_ore_code(code)
                          for bid, code in mapping.items()
                          if code and code.startswith("ore")}
            keep_ids = {bid for bid, ot in ore_parsed.items()
                        if ot.metal in tracked}
            ore_types = {bid: (ot.metal, ot.label, ot.grade)
                         for bid, ot in ore_parsed.items() if bid in keep_ids}
            # saltpeter is not an ore- block; it forms on cave walls as
            # "saltpeter-*". Track it too if selected.
            if "saltpeter" in tracked:
                for bid, code in mapping.items():
                    if code and (code == "saltpeter" or code.startswith("saltpeter-")):
                        keep_ids.add(bid)
                        ore_types[bid] = ("saltpeter", "Saltpeter", "")
            # meteoric iron is the meteorite core block "meteorite-iron"
            # (a fallen meteorite), not a normal ore- vein
            if "meteoriciron" in tracked:
                for bid, code in mapping.items():
                    if code == "meteorite-iron":
                        keep_ids.add(bid)
                        ore_types[bid] = ("meteoriciron", "Meteoric iron", "")
            # fireclay is a clay deposit (rawclay-fire-*), not an ore- block
            if "fireclay" in tracked:
                for bid, code in mapping.items():
                    if code and code.startswith("rawclay-fire"):
                        keep_ids.add(bid)
                        ore_types[bid] = ("fireclay", "Fireclay", "")
            # olivine = harvestable crystal clusters (blast-furnace steel flux)
            if "olivine" in tracked:
                for bid, code in mapping.items():
                    if code and code.startswith("crystal-olivine"):
                        keep_ids.add(bid)
                        ore_types[bid] = ("olivine", "Olivine", "")
            # misc surface finds (also not ore- blocks)
            if "beehive" in tracked:
                for bid, code in mapping.items():
                    if code and code.startswith("wildbeehive"):
                        keep_ids.add(bid)
                        ore_types[bid] = ("beehive", "Wild beehive", "")
            if "resin" in tracked:
                for bid, code in mapping.items():
                    # log-resin-*  (unharvested); excludes log-resinharvested-*
                    if code and code.startswith("log-resin-"):
                        keep_ids.add(bid)
                        ore_types[bid] = ("resin", "Tree resin", "")
            keep_arr = np.array(sorted(keep_ids), dtype=np.int32)

            # tree species lookup: block id -> species index+1 (0 = not a log)
            nsp = len(TREE_SPECIES)
            maxid = max(mapping) if mapping else 1
            log_species = np.zeros(maxid + 1, dtype=np.int16)
            for bid, code in mapping.items():
                if code and code.startswith("log-grown-"):
                    sp = code.split("-")[2]
                    if sp in TREE_SPECIES:
                        log_species[bid] = TREE_SPECIES.index(sp) + 1

            # rock type lookup: block id -> rock index+1 (0 = not a rock-<type>)
            nrock = len(ROCK_TYPES)
            rock_idx = {r: i for i, r in enumerate(ROCK_TYPES)}
            rock_type = np.zeros(maxid + 1, dtype=np.int16)
            for bid, code in mapping.items():
                if code and code.startswith("rock-") and code.count("-") == 1:
                    rt = code.split("-")[1]
                    if rt in rock_idx:
                        rock_type[bid] = rock_idx[rt] + 1

            # gravel / sand lookups: block id -> source rock index+1 (by variant,
            # e.g. gravel-andesite, sand-basalt). Same variant set as rocks.
            gravel_type = np.zeros(maxid + 1, dtype=np.int16)
            sand_type = np.zeros(maxid + 1, dtype=np.int16)
            for bid, code in mapping.items():
                if not code:
                    continue
                if code.startswith("gravel-"):
                    rt = code.split("-")[1]
                    if rt in rock_idx:
                        gravel_type[bid] = rock_idx[rt] + 1
                elif code.startswith("sand-"):
                    rt = code.split("-")[1]
                    if rt in rock_idx:
                        sand_type[bid] = rock_idx[rt] + 1

            prev_trees = prev["trees"] if (prev and "trees" in prev) else {}
            new_trees: dict[int, np.ndarray] = {}
            prev_rocks = prev["rocks"] if (prev and "rocks" in prev) else {}
            new_rocks: dict[int, np.ndarray] = {}
            prev_gravel = prev["gravel"] if (prev and "gravel" in prev) else {}
            new_gravel: dict[int, np.ndarray] = {}
            prev_sand = prev["sand"] if (prev and "sand" in prev) else {}
            new_sand: dict[int, np.ndarray] = {}

            # raw-clay lookup: block id -> 1 blue / 2 red / 0 neither
            clay_type = np.zeros(maxid + 1, dtype=np.int16)
            for bid, code in mapping.items():
                if not code:
                    continue
                if code.startswith("rawclay-blue"):
                    clay_type[bid] = 1
                elif code.startswith("rawclay-red"):
                    clay_type[bid] = 2
            prev_clay = prev["clay"] if (prev and "clay" in prev) else {}
            new_clay: dict[int, np.ndarray] = {}

            total = con.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
            new_hashes: dict[int, int] = {}
            new_ore: dict[int, tuple] = {}
            decoded = reused = 0
            done = 0
            for pos, data in con.execute("SELECT position, data FROM chunk"):
                done += 1
                if progress and done % 1000 == 0:
                    progress(done, total, f"{decoded:,} new / {reused:,} cached")
                # hash only the block layer (field 1); lighting/entity re-saves don't
                # change ore, so this lets unchanged chunks be reused on rescan.
                i = 0
                _t, i = _varint(data, i)
                ln, i = _varint(data, i)
                field1 = bytes(data[i:i + ln]) if ln else b""
                crc = zlib.crc32(field1) & 0xFFFFFFFF
                new_hashes[pos] = crc
                if prev_hashes.get(pos) == crc:
                    reused += 1
                    if pos in prev_ore:
                        new_ore[pos] = prev_ore[pos]
                    if pos in prev_trees:
                        new_trees[pos] = prev_trees[pos]
                    if pos in prev_rocks:
                        new_rocks[pos] = prev_rocks[pos]
                    if pos in prev_gravel:
                        new_gravel[pos] = prev_gravel[pos]
                    if pos in prev_sand:
                        new_sand[pos] = prev_sand[pos]
                    if pos in prev_clay:
                        new_clay[pos] = prev_clay[pos]
                    continue
                decoded += 1
                if ln == 0:
                    continue
                arr = self._decode(field1)
                mask = np.isin(arr, keep_arr)
                idxs = np.nonzero(mask)[0].astype(np.uint16)
                if idxs.size:
                    new_ore[pos] = (idxs, arr[idxs].astype(np.uint16))
                # tally tree logs per species in this chunk (one bincount)
                tc = np.bincount(log_species[arr], minlength=nsp + 1)[1:]
                if tc.any():
                    new_trees[pos] = tc.astype(np.int32)
                # tally rock blocks per type in this chunk (one bincount)
                rc = np.bincount(rock_type[arr], minlength=nrock + 1)[1:]
                if rc.any():
                    new_rocks[pos] = rc.astype(np.int32)
                # tally gravel & sand per source rock type (one bincount each)
                gc = np.bincount(gravel_type[arr], minlength=nrock + 1)[1:]
                if gc.any():
                    new_gravel[pos] = gc.astype(np.int32)
                sdc = np.bincount(sand_type[arr], minlength=nrock + 1)[1:]
                if sdc.any():
                    new_sand[pos] = sdc.astype(np.int32)
                # tally blue & red raw clay in this chunk (one bincount)
                cc = np.bincount(clay_type[arr], minlength=3)[1:]
                if cc.any():
                    new_clay[pos] = cc.astype(np.int32)

            spawn = self._read_spawn(con)

            # terrain overlay PNG (explored area) while the snapshot is open
            overlay = None
            try:
                if progress:
                    progress(total, total, "rendering terrain map…")
                import map_overlay
                os.makedirs(ASSETS_DIR, exist_ok=True)
                key = "%08x" % (zlib.crc32(save_path.encode("utf8")) & 0xFFFFFFFF)
                png = os.path.join(ASSETS_DIR, f"overlay_{key}.png")
                info = map_overlay.build_overlay(con, png)
                if info:
                    info["url"] = f"/overlay_{key}.png"
                    overlay = info
            except Exception:
                overlay = None
            con.close()

            # per-metal block totals, and the drop vs the previous scan so the
            # app can report veins that were mined out since last time
            bid_metal = {bid: mt[0] for bid, mt in ore_types.items()}
            metal_counts: dict[str, int] = {}
            for _idxs, bids in new_ore.values():
                vals, cnts = np.unique(bids, return_counts=True)
                for v, c in zip(vals, cnts):
                    mt = bid_metal.get(int(v))
                    if mt:
                        metal_counts[mt] = metal_counts.get(mt, 0) + int(c)
            prev_counts = prev.get("metal_counts", {}) if prev else {}
            mined_delta = {m: prev_counts[m] - metal_counts.get(m, 0)
                           for m in prev_counts
                           if prev_counts[m] - metal_counts.get(m, 0) > 0}

            cache = dict(
                version=CACHE_VERSION, save_path=save_path,
                cache_file=cache_path(save_path), scanned_at=time.time(),
                cluster_size=cluster_size, hashes=new_hashes, ore=new_ore,
                ore_types=ore_types, overlay=overlay, spawn=spawn,
                trees=new_trees, tree_species=TREE_SPECIES,
                rocks=new_rocks, rock_types=ROCK_TYPES,
                gravel=new_gravel, sand=new_sand, clay=new_clay,
                metal_counts=metal_counts, mined_delta=mined_delta,
                stats=dict(total_chunks=total, decoded=decoded, reused=reused,
                           ore_blocks=sum(len(v[0]) for v in new_ore.values())))
            if progress:
                progress(total, total, "clustering…")
            cache["clusters"] = cluster_from_cache(cache, cluster_size)
            cache["stats"]["clusters"] = len(cache["clusters"])
            save_cache(cache)
            return cache
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    sc = Scanner()
    saves = find_saves()
    path = sys.argv[1] if len(sys.argv) > 1 else saves[0][1]
    print("Scanning:", path)
    t0 = time.time()
    cache = sc.scan_to_cache(
        path, progress=lambda d, t, m: print(f"  {d}/{t}  {m}", end="\r"))
    print(f"\nDone in {time.time()-t0:.1f}s  stats={cache['stats']}")
    print("cache file:", cache["cache_file"])
