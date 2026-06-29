# Vintage Story Ore Finder

Find the **exact location of every valuable ore** in your Vintage Story world — down
to the block.

It reads your save file and decodes the world using Vintage Story's *own* code, so the
coordinates are real block positions: no heatmaps, no guessing. It only sees parts of
the world you've actually explored (chunks the game has generated).

> **Unofficial fan tool.** Not affiliated with or endorsed by the Vintage Story
> developers. You need your own copy of the game installed — the app borrows the
> game's decoder at runtime and **no game files are bundled or redistributed**.

---

## Features

- **Precise ore targeting** — exact X / Y / Z for every ore block, grouped into veins.
- **30+ materials** — all metals (gold, silver, **meteoric iron**, iron, copper, …),
  minerals (borax, saltpeter, fireclay, sulfur, …), coal, gems, and surface finds
  (wild beehives, tree resin). Toggle each on/off; metals are ordered by value.
- **Terrain map** — a shaded-relief backdrop built from the explored area so the dots
  sit on a recognisable map.
- **Density overlays** — heatmaps for **trees** (by species), **rock layers**
  (limestone, marble, …), **gravel** and **sand** (by rock type), and **clay**
  (blue / red). Great for finding industry materials and building blocks.
- **Custom markers** — drop Base / Mined / Note pins, give them **custom names and
  colours**, and use any pin as the coordinate origin.
- **Fast incremental rescans** — only re-reads chunks that changed since last time, and
  reports what was **mined out** since the previous scan. Optional auto-rescan keeps the
  map current on a timer.
- **Read-only & safe** — works on a temporary copy of your save; it never modifies the
  live game file.

---

## Requirements

- **Windows** with **Vintage Story** installed (the app loads the game's DLLs to decode
  the world).
- **Python 3.14 or newer** (the app uses the standard-library `compression.zstd`, added
  in 3.14).

## Install & run (from source)

```bash
git clone https://github.com/atesayaksu-sudo/Vintage-Story-Material-Scanner.git
cd Vintage-Story-Material-Scanner
pip install -r requirements.txt
python app.py
```

On Windows you can also just double-click **`launch.bat`**.

---

## Quick start

1. Pick your world from the **World save** dropdown (top right).
2. Click **Scan world** — the first scan takes ~1.5–2 minutes (it copies your save and
   reads every explored chunk). Later scans are much faster.
3. The map fills with coloured dots, one colour per metal. **Scroll** to zoom, **drag**
   to pan, **⊡** to fit everything. Click a dot (or a row in the right-hand list) to see
   its exact coordinates.
4. Use the left panel to toggle materials, switch on a density overlay, or place markers.

Coordinates are shown **relative to your world spawn** by default, matching the in-game
position readout. You can re-base them on any marker.

> **Tip:** Vintage Story only writes changes to the save on autosave or
> *Save and Quit to Title*. If you just mined something and it still shows up, save the
> world that way first, then rescan.

---

## How it works

The save (`.vcdbs`) is a SQLite database. Each chunk's block data is a protobuf payload
containing a Zstandard-compressed block layer. Rather than reverse-engineer the
(versioned) bit-packing, the app calls Vintage Story's own `UnpackBlocksTo` through
[pythonnet](https://github.com/pythonnet/pythonnet), so decoding always matches the
game exactly. Block IDs are mapped to codes from the save's `gamedata` table, and the
results are cached so rescans only touch chunks that changed.

## License

[MIT](LICENSE) © 2026 Ateş Ay Aksu
