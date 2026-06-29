"""
Vintage Story Ore Finder - desktop app.

Top-down map of every valuable ore deposit in your world, decoded straight from
the save file using the game's own chunk decoder. Results are cached to disk:
the app opens showing your last scan instantly, and "Rescan" only re-reads chunks
whose data changed since last time.
"""

from __future__ import annotations
import os
import json
import time
import asyncio
import logging
import traceback

import numpy as np
import flet as ft
import flet.canvas as cv

import scanner
import map_overlay
from scanner import (Scanner, find_saves, find_game_dir, load_cache,
                     cluster_from_cache, DEFAULT_METALS, OreCluster, GRADE_RANK,
                     load_settings, save_settings, is_game_dir)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orefinder.log")
logging.basicConfig(
    filename=LOG_PATH, level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orefinder")

# materials grouped by category for the toggle panel: (key, label, colour)
CATALOG = {
    "Metals": [
        ("gold", "Gold", "#FFD24A"), ("silver", "Silver", "#D8E1EC"),
        ("lead/silver", "Galena (lead/silver)", "#9FB2C4"),
        ("meteoriciron", "Meteoric iron", "#8E9BD4"),
        ("iron", "Iron", "#E36A4B"), ("copper", "Copper", "#E08A3C"),
        ("tin", "Tin", "#C8D6E5"), ("zinc", "Zinc", "#B0A6C9"),
        ("nickel", "Nickel", "#8FBF9F"), ("titanium", "Titanium", "#7FA8C9"),
        ("chromium", "Chromium", "#6FB6B0"), ("bismuth", "Bismuth", "#C98FB0"),
        ("manganese", "Manganese", "#A98FC9"), ("mercury", "Cinnabar (mercury)", "#FF8FA0"),
        ("uranium", "Uranium", "#BFE34B"),
    ],
    "Minerals": [
        ("borax", "Borax", "#E8D27A"), ("sulfur", "Sulfur", "#E8E04A"),
        ("saltpeter", "Saltpeter", "#CFE8B0"), ("alum", "Alum", "#D0C0E8"),
        ("fluorite", "Fluorite", "#7AE8C8"), ("phosphorus", "Phosphorite", "#C8A07A"),
        ("potash", "Sylvite (potash)", "#C8E07A"), ("graphite", "Graphite", "#8A8A96"),
        ("salt", "Halite (salt)", "#E0E8F0"), ("fireclay", "Fireclay", "#C9A876"),
    ],
    "Coal": [
        ("anthracite", "Anthracite", "#A6ADBD"), ("bituminous", "Bituminous coal", "#717A8C"),
        ("lignite", "Lignite", "#9A7E5E"),
    ],
    "Gems": [
        ("diamond", "Diamond", "#9FE8FF"), ("emerald", "Emerald", "#5FD08A"),
        ("ruby", "Ruby", "#FF6B7A"), ("sapphire", "Sapphire", "#6B8CFF"),
        ("peridot", "Peridot", "#AEDD5F"), ("lapis", "Lapis lazuli", "#5A7AE8"),
    ],
    "Misc": [
        ("beehive", "Wild beehive", "#FF9E2C"), ("resin", "Tree resin", "#9C5A2E"),
    ],
}
METAL_COLORS = {key: col for items in CATALOG.values() for key, _, col in items}
MATERIAL_LABEL = {key: lbl for items in CATALOG.values() for key, lbl, _ in items}

# user map markers: kind -> colour
MARKER_STYLE = {"base": "#FFE15C", "mined": "#8A96A6", "note": "#5FD0FF"}
MARKER_KINDS = {"Base": "base", "Mined area": "mined", "Note": "note"}
# swatches for the per-marker colour picker
MARKER_PALETTE = ["#FFE15C", "#FFB02E", "#FF6B6B", "#FF8FB1", "#C77DFF",
                  "#7B8CFF", "#5FD0FF", "#3DDC97", "#A3E635", "#E8EDF4",
                  "#9AA6B6", "#5A6478"]


def _ago(ts: float) -> str:
    s = max(0, int(time.time() - ts))
    if s < 60: return f"{s}s ago"
    if s < 3600: return f"{s//60}m ago"
    if s < 86400: return f"{s//3600}h ago"
    return f"{s//86400}d ago"


class OreFinderApp:
    DRAW_CAP = 4000

    def __init__(self, page: ft.Page):
        self.page = page
        self.settings = load_settings()
        # a custom Vintage Story install folder, if the user pointed us at one
        self.game_dir = self.settings.get("game_dir") or None
        self.scanner: Scanner | None = None
        self.cache: dict | None = None
        self.clusters: list = []
        self.filtered: list = []
        self.draw_order: list = []
        self.selected = None
        self.busy = False

        self.scale = 0.02
        self.view_cx = 512000.0
        self.view_cz = 512000.0
        self.canvas_w = 940
        self.canvas_h = 720

        self.enabled_metals: set = set()   # all materials off by default
        self.origin_x = 0
        self.origin_z = 0

        # vectorised draw arrays (rebuilt in _apply_filter)
        self._ax = np.zeros(0)
        self._az = np.zeros(0)
        self._ar = np.zeros(0)
        self._acol: list = []

        # user markers
        self.markers: list = []        # [{x,z,kind,label}]
        self._editing = None           # marker dict currently being renamed
        self._coloring = None          # marker dict currently picking a colour
        self._clear_armed = False      # two-step confirm for "Remove markers"
        self.place_mode = False

        # redraw coalescing (keeps drag/zoom responsive)
        self._redraw_pending = False
        self._redraw_delay = 0.012
        self._hover_idx = -1
        self.show_overlay = True
        self._overlay_img = None        # reused cv.Image (avoids reloading PNG)
        self._did_fit = False           # fit once the real canvas size is known
        # density overlays (trees / rocks / gravel / sand) — each replaces terrain
        # while on (one image -> same perf); registry built in _build
        self.density_layers = {}
        self.auto_rescan = False
        self.auto_interval_sec = 300
        self._auto_running = False
        self.list_sort = "biggest"
        # terrain is hidden while actively panning/zooming (it's expensive to
        # re-render at high zoom) and snaps back when interaction stops
        self._interacting = False
        self._idle_scheduled = False
        self._last_interact = 0.0

        page.on_error = lambda e: log.error("PAGE ERROR: %s", getattr(e, "data", e))
        self._build()
        self.page.run_task(self._startup)

    # ------------------------------------------------------------------ UI
    def _build(self):
        p = self.page
        p.title = "Vintage Story Ore Finder"
        p.theme_mode = ft.ThemeMode.DARK
        p.bgcolor = "#10131A"
        p.padding = 0

        saves = self._all_saves()
        self.save_dd = ft.Dropdown(
            label="World save", value=saves[0][1] if saves else None,
            options=[ft.dropdown.Option(key=pth, text=f"{nm}  ({sz//1_000_000} MB)")
                     for nm, pth, sz in saves],
            width=300, dense=True, filled=True, on_select=self._on_save_change)
        # file pickers for worlds outside the default folder + custom game install
        self.save_picker = ft.FilePicker()
        self.gamedir_picker = ft.FilePicker()
        p.services.extend([self.save_picker, self.gamedir_picker])
        self.browse_btn = ft.IconButton(
            ft.Icons.FOLDER_OPEN, tooltip="Browse for a world save (.vcdbs) "
            "anywhere on your PC", on_click=self._on_browse_save)
        self.refresh_btn = ft.FilledButton(
            "Rescan", icon=ft.Icons.RADAR, on_click=self._on_refresh)
        self.progress = ft.ProgressBar(width=240, value=0, visible=False)
        self.status = ft.Text("Ready.", size=12, color="#8A94A6")

        self.metal_switches = {}
        chips = []
        for category, items in CATALOG.items():
            chips.append(ft.Row([
                ft.Text(category.upper(), size=10, weight=ft.FontWeight.BOLD,
                        color="#5A6478"),
                ft.Container(expand=True),
                ft.TextButton(category, scale=0.7, style=ft.ButtonStyle(padding=0),
                              tooltip=f"Toggle all {category}",
                              on_click=lambda e, items=items: self._toggle_group(items))],
                spacing=2, tight=True))
            # Metals stay in importance order; other categories alphabetical
            shown = items if category == "Metals" else sorted(
                items, key=lambda t: t[1].lower())
            for key, label, col in shown:
                sw = ft.Switch(value=False, active_color=col, scale=0.65,
                               on_change=self._on_filter)
                self.metal_switches[key] = sw
                chips.append(ft.Row([
                    sw, ft.Container(width=11, height=11, bgcolor=col, border_radius=3),
                    ft.Text(label, size=12)], spacing=4, tight=True))
        metals_panel = ft.Column(chips, spacing=0, scroll=ft.ScrollMode.AUTO,
                                 height=240)
        all_row = ft.Row([
            ft.TextButton("All", on_click=lambda e: self._set_all_metals(True)),
            ft.TextButton("None", on_click=lambda e: self._set_all_metals(False))],
            spacing=2)

        self.cluster_slider = ft.Slider(
            min=4, max=32, value=12, divisions=7, label="{value} blocks", width=220,
            on_change_end=self._on_merge_change)
        self.merge_count_txt = ft.Text("", size=11, color="#5A6478")
        self.origin_x_f = ft.TextField(label="Origin X", value="0", width=105,
                                       dense=True, on_submit=self._on_origin)
        self.origin_z_f = ft.TextField(label="Origin Z", value="0", width=105,
                                       dense=True, on_submit=self._on_origin)
        self.stats_txt = ft.Text("", size=12, color="#8A94A6")

        # markers controls
        self.marker_kind_dd = ft.Dropdown(
            label="Type", value="Base", width=130, dense=True,
            options=[ft.dropdown.Option(k) for k in MARKER_KINDS])
        self.marker_label_f = ft.TextField(label="Name", width=110, dense=True,
                                            hint_text="optional")
        self.place_btn = ft.OutlinedButton("Place marker on map",
                                           icon=ft.Icons.ADD_LOCATION_ALT,
                                           on_click=self._toggle_place)
        self.clear_markers_btn = ft.TextButton(
            "Remove markers", icon=ft.Icons.DELETE_SWEEP,
            tooltip="Removes every marker except Base markers",
            on_click=self._on_clear_markers)
        self.markers_list = ft.Column([], spacing=2)

        # map-overlay layer controls (used in the left "Map overlays" section)
        self.terrain_switch = ft.Switch(value=True, scale=0.7,
                                        on_change=self._on_overlay_toggle)
        # density overlays — registry of identical layers (trees/rocks/gravel/sand)
        self._make_density_layer("trees", "Trees", "trees", scanner.TREE_SPECIES,
                                 "All trees", map_overlay.build_tree_density)
        self._make_density_layer("rocks", "Rocks", "rocks", scanner.ROCK_TYPES,
                                 "All rock", map_overlay.build_rock_density)
        self._make_density_layer("gravel", "Gravel", "gravel", scanner.ROCK_TYPES,
                                 "All gravel", map_overlay.build_gravel_density)
        self._make_density_layer("sand", "Sand", "sand", scanner.ROCK_TYPES,
                                 "All sand", map_overlay.build_sand_density)
        self._make_density_layer("clay", "Clay", "clay", scanner.CLAY_TYPES,
                                 "All clay", map_overlay.build_clay_density)

        # auto-rescan controls
        self.auto_switch = ft.Switch(value=False, scale=0.7,
                                     on_change=self._on_autorescan_toggle)
        self.auto_interval_dd = ft.Dropdown(
            value="5", width=95, dense=True, on_select=self._on_auto_interval,
            options=[ft.dropdown.Option(m) for m in ("2", "5", "10", "15")])

        left = ft.Container(
            width=280, bgcolor="#161A23", padding=16,
            content=ft.Column([
                ft.Row([ft.Text("Materials", weight=ft.FontWeight.BOLD, size=13),
                        ft.Container(expand=True), all_row]),
                ft.Text("All off by default — switch on what you want.",
                        size=10, color="#5A6478"),
                metals_panel,
                ft.Divider(height=16, color="#222838"),
                ft.Text("Map overlays", weight=ft.FontWeight.BOLD, size=13),
                ft.Row([self.terrain_switch, ft.Text("Terrain", size=12)],
                       spacing=4, tight=True),
                *[ft.Row([L["switch"], ft.Text(L["label"], size=12),
                          ft.Container(expand=True), L["dropdown"]],
                         vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=4)
                  for L in self.density_layers.values()],
                ft.Text("Density overlays replace terrain while on (one layer at "
                        "a time keeps it smooth).", size=10, color="#5A6478"),
                ft.Divider(height=16, color="#222838"),
                ft.Row([ft.Text("Vein grouping", size=12, color="#8A94A6"),
                        ft.Icon(ft.Icons.HELP_OUTLINE, size=13, color="#5A6478",
                                tooltip="Ore blocks within this many blocks of each "
                                "other are grouped into one vein. Higher = fewer, "
                                "bigger dots (whole deposits); lower = more, smaller "
                                "dots. Bigger veins draw as bigger dots.")]),
                self.cluster_slider,
                self.merge_count_txt,
                ft.Divider(height=16, color="#222838"),
                ft.Text("Markers", weight=ft.FontWeight.BOLD, size=13),
                ft.Row([self.marker_kind_dd, self.marker_label_f], spacing=8),
                ft.Row([self.place_btn, self.clear_markers_btn], spacing=6,
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                self.markers_list,
                ft.Divider(height=16, color="#222838"),
                ft.Text("Coordinate origin (defaults to spawn)", size=12,
                        color="#8A94A6"),
                ft.Row([self.origin_x_f, self.origin_z_f], spacing=8),
                ft.Text("Set to spawn so coords match your in-game position.",
                        size=10, color="#5A6478"),
                ft.Divider(height=16, color="#222838"),
                ft.Text("Auto-rescan", weight=ft.FontWeight.BOLD, size=13),
                ft.Row([self.auto_switch, ft.Text("On", size=12),
                        ft.Container(expand=True),
                        self.auto_interval_dd, ft.Text("min", size=11)],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Text("Re-reads changed chunks on a timer; ore you've mined "
                        "disappears automatically.", size=10, color="#5A6478"),
                ft.Divider(height=16, color="#222838"),
                self.stats_txt,
            ], spacing=8, scroll=ft.ScrollMode.AUTO, expand=True))

        self.canvas = cv.Canvas(shapes=[], expand=True,
                                on_resize=self._on_canvas_resize)
        self.map_gd = ft.GestureDetector(
            expand=True,
            content=ft.Container(self.canvas, expand=True, bgcolor="#0B0E14",
                                 border=ft.Border.all(1, "#222838")),
            on_tap_down=self._on_tap, on_pan_update=self._on_pan,
            on_scroll=self._on_scroll, on_hover=self._on_hover,
            hover_interval=60)
        self.center_x_f = ft.TextField(label="X", width=90, dense=True,
                                       on_submit=self._on_center)
        self.center_z_f = ft.TextField(label="Z", width=90, dense=True,
                                       on_submit=self._on_center)
        self.hover_label = ft.Text("Hover a vein to see its coordinates",
                                   size=12, color="#8A94A6")
        zoom_row = ft.Row([
            ft.IconButton(ft.Icons.ADD, on_click=lambda e: self._zoom(1.4)),
            ft.IconButton(ft.Icons.REMOVE, on_click=lambda e: self._zoom(1 / 1.4)),
            ft.IconButton(ft.Icons.FIT_SCREEN, tooltip="Fit all",
                          on_click=lambda e: (self._fit(), self._redraw())),
            ft.Container(width=10),
            self.center_x_f, self.center_z_f,
            ft.FilledTonalButton("Go to", icon=ft.Icons.MY_LOCATION,
                                 on_click=self._on_center),
            ft.Container(width=10),
            self.hover_label],
            spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        center = ft.Container(
            padding=12, expand=True,
            content=ft.Column([zoom_row, self.map_gd], spacing=6, expand=True))

        self.detail = ft.Container(
            padding=14, bgcolor="#161A23", border_radius=10,
            content=ft.Text("Click a deposit on the map or in the list.",
                            size=12, color="#8A94A6"))
        self.list_view = ft.ListView(expand=True, spacing=4, padding=4)
        self.sort_dd = ft.Dropdown(
            value="Biggest", width=170, dense=True, on_select=self._on_sort,
            options=[ft.dropdown.Option(o) for o in
                     ("Biggest", "Nearest base", "Best value (near + rich)")])
        right = ft.Container(
            width=340, bgcolor="#121620", padding=14,
            content=ft.Column([
                ft.Text("Deposit details", weight=ft.FontWeight.BOLD, size=13),
                self.detail,
                ft.Divider(height=14, color="#222838"),
                ft.Row([ft.Text("Deposits (by area)", weight=ft.FontWeight.BOLD,
                                size=13), ft.Container(expand=True),
                        ft.Text("Sort:", size=11, color="#8A94A6"), self.sort_dd],
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.list_view], spacing=10, expand=True))

        # custom title bar (native one is hidden in main()) - drag the title area
        # to move the window; window controls on the far right
        title_drag = ft.WindowDragArea(
            expand=True,
            content=ft.Container(
                padding=ft.Padding.only(left=16),
                content=ft.Row([
                    ft.Icon(ft.Icons.TRAVEL_EXPLORE, color="#FFD24A"),
                    ft.Text("Vintage Story Ore Finder",
                            weight=ft.FontWeight.BOLD, size=16),
                    ft.Container(expand=True)])))
        win_btns = ft.Row(spacing=0, controls=[
            ft.IconButton(ft.Icons.REMOVE, icon_size=18, icon_color="#8A94A6",
                          tooltip="Minimize", on_click=self._win_min),
            ft.IconButton(ft.Icons.CROP_SQUARE, icon_size=14, icon_color="#8A94A6",
                          tooltip="Maximize", on_click=self._win_max),
            ft.IconButton(ft.Icons.CLOSE, icon_size=18, icon_color="#E36A4B",
                          tooltip="Close", on_click=self._win_close)])
        header = ft.Container(
            bgcolor="#161A23", padding=ft.Padding.symmetric(vertical=8, horizontal=0),
            content=ft.Row([
                title_drag,
                self.save_dd, self.browse_btn, self.refresh_btn,
                ft.Column([self.progress, self.status], spacing=2, tight=True),
                ft.Container(width=8), win_btns],
                spacing=14, vertical_alignment=ft.CrossAxisAlignment.CENTER))

        p.add(ft.Column([header, ft.Row([left, center, right], spacing=0, expand=True)],
                        spacing=0, expand=True))

        if not find_game_dir():
            self._set_status("⚠ Vintage Story install not found.", err=True)

    # --------------------------------------------------------------- startup
    async def _startup(self):
        """Load the cached scan for the selected save, if any."""
        save = self.save_dd.value
        if not save:
            self._set_status("No worlds found in the default folder — click the "
                             "📂 folder icon to browse for your .vcdbs save.")
            return
        self._set_status("Loading saved scan…")
        self._load_markers()
        self._refresh_markers()
        cache = await asyncio.to_thread(load_cache, save)
        if cache:
            self.cache = cache
            self.clusters = cache.get("clusters", [])
            self.cluster_slider.value = cache.get("cluster_size", 12)
            sp = cache.get("spawn")
            if sp:           # default origin to world spawn => coords match in-game
                self.origin_x, self.origin_z = int(sp[0]), int(sp[1])
                self.origin_x_f.value = str(int(sp[0]))
                self.origin_z_f.value = str(int(sp[1]))
            self._apply_filter()
            self._fit()
            self._redraw()
            st = cache.get("stats", {})
            self._set_status(
                f"Loaded cache · {st.get('clusters', len(self.clusters)):,} deposits · "
                f"scanned {_ago(cache.get('scanned_at', time.time()))}. "
                f"Rescan only reads changed chunks.")
        else:
            self._set_status("No saved scan yet — click Rescan to scan this world.")

    async def _on_save_change(self, e):
        await self._startup()

    # --------------------------------------------------- save / game discovery
    def _all_saves(self):
        """Saves in the default folder plus any the user browsed to elsewhere."""
        found = find_saves()
        seen = {p for _, p, _ in found}
        for p in self.settings.get("extra_saves", []):
            if p not in seen and os.path.exists(p):
                nm = os.path.splitext(os.path.basename(p))[0]
                found.append((nm, p, os.path.getsize(p)))
                seen.add(p)
        return found

    def _rebuild_save_dd(self, select=None):
        saves = self._all_saves()
        self.save_dd.options = [
            ft.dropdown.Option(key=p, text=f"{nm}  ({sz//1_000_000} MB)")
            for nm, p, sz in saves]
        if select:
            self.save_dd.value = select
        self.page.update()

    async def _on_browse_save(self, e):
        files = await self.save_picker.pick_files(
            dialog_title="Select a Vintage Story world save",
            allow_multiple=False, allowed_extensions=["vcdbs"])
        if not files:
            return
        path = files[0].path
        if not path or not path.lower().endswith(".vcdbs"):
            self._set_status("That isn't a .vcdbs world save.", err=True)
            return
        extra = self.settings.get("extra_saves", [])
        if path not in extra:
            extra.append(path)
            self.settings["extra_saves"] = extra
            save_settings(self.settings)
        self._rebuild_save_dd(select=path)
        self._set_status(f"Loaded save: {os.path.basename(path)}")
        await self._startup()

    async def _on_browse_gamedir(self):
        path = await self.gamedir_picker.get_directory_path(
            dialog_title="Select your Vintage Story install folder "
            "(contains VintagestoryLib.dll)")
        if not path:
            return
        if not is_game_dir(path):
            self._set_status("That folder doesn't contain VintagestoryLib.dll — "
                             "pick your Vintage Story install folder.", err=True)
            return
        self.game_dir = path
        self.settings["game_dir"] = path
        save_settings(self.settings)
        self.scanner = None          # force re-init against the new game folder
        self._set_status("Game folder set ✓ — click Rescan.")

    # ----------------------------------------------------------- transform
    def w2s(self, wx, wz):
        return ((wx - self.view_cx) * self.scale + self.canvas_w / 2,
                (wz - self.view_cz) * self.scale + self.canvas_h / 2)

    def s2w(self, sx, sy):
        return ((sx - self.canvas_w / 2) / self.scale + self.view_cx,
                (sy - self.canvas_h / 2) / self.scale + self.view_cz)

    # ------------------------------------------------------------- drawing
    def _redraw(self):
        try:
            shapes = []
            ov = self.cache.get("overlay") if self.cache else None
            # a density overlay (tree/rock/gravel/sand) replaces terrain when on —
            # it has terrain baked in, so it's still one image (stays smooth)
            dens_img = dens_bounds = None
            for L in self.density_layers.values():
                if L["mode"] and L["img"] and L["bounds"]:
                    dens_img, dens_bounds = L["img"], L["bounds"]
                    break
            show_dens = dens_img is not None
            show_ov = bool(ov and self.show_overlay and not show_dens)
            if show_dens:
                dx0, dz0, dw, dh = dens_bounds
                ix, iy = self.w2s(dx0, dz0)
                dens_img.x, dens_img.y = ix, iy
                dens_img.width = dw * self.scale
                dens_img.height = dh * self.scale
                shapes.append(dens_img)
            if show_ov:
                ix, iy = self.w2s(ov["x0"], ov["z0"])
                img = self._overlay_img
                if img is None or img.src != ov["url"]:
                    img = cv.Image(src=ov["url"])
                    self._overlay_img = img
                img.x, img.y = ix, iy
                img.width, img.height = ov["w"] * self.scale, ov["h"] * self.scale
                shapes.append(img)

            if not show_ov and not show_dens:
                # only draw the reference grid when there's no terrain backdrop
                gridpaint = ft.Paint(color="#1A2030", stroke_width=1)
                wx0, wz0 = self.s2w(0, 0)
                wx1, wz1 = self.s2w(self.canvas_w, self.canvas_h)
                span = max(wx1 - wx0, wz1 - wz0)
                step = 500
                for s in (100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000):
                    if span / s <= 24:
                        step = s
                        break
                gx = int(wx0 // step) * step
                while gx < wx1:
                    sx, _ = self.w2s(gx, 0)
                    shapes.append(cv.Line(sx, 0, sx, self.canvas_h, gridpaint))
                    gx += step
                gz = int(wz0 // step) * step
                while gz < wz1:
                    _, sy = self.w2s(0, gz)
                    shapes.append(cv.Line(0, sy, self.canvas_w, sy, gridpaint))
                    gz += step

            # Fully-vectorised: transform all veins, cull off-screen, then collapse
            # to one dot per 3px cell (keeping the highest-priority vein per cell)
            # and cap. All heavy work is numpy; only the final <=CAP shapes are
            # built in python -> redraw stays ~milliseconds even at 287k veins.
            n = self._ax.shape[0]
            if n:
                sxx = (self._ax - self.view_cx) * self.scale + self.canvas_w / 2
                syy = (self._az - self.view_cz) * self.scale + self.canvas_h / 2
                vis = np.nonzero((sxx >= -6) & (sxx <= self.canvas_w + 6)
                                 & (syy >= -6) & (syy <= self.canvas_h + 6))[0]
                if vis.size:
                    gx = np.floor(sxx[vis] / 5).astype(np.int64) + 4
                    gy = np.floor(syy[vis] / 5).astype(np.int64) + 4
                    key = gx * 100003 + gy
                    # first occurrence per cell = highest-priority vein (vis is
                    # already in round-robin priority order)
                    _, first = np.unique(key, return_index=True)
                    first.sort()
                    chosen = vis[first[:self.DRAW_CAP]]
                    xs = sxx[chosen].tolist()
                    ys = syy[chosen].tolist()
                    cj = chosen.tolist()
                    acol = self._acol
                    asize = self._asize
                    # filled dots, sized by vein block-count, batched into one
                    # Points control per (colour, size) so it stays fast
                    groups: dict = {}
                    for k in range(len(cj)):
                        j = cj[k]
                        groups.setdefault((acol[j], int(asize[j])), []).append(
                            ft.Offset(xs[k], ys[k]))
                    for (col, sz), pts in groups.items():
                        shapes.append(cv.Points(
                            points=pts, point_mode=cv.PointMode.POINTS,
                            paint=ft.Paint(color=col, stroke_width=sz,
                                           stroke_cap=ft.StrokeCap.ROUND)))
            # user markers (always drawn)
            for m in self.markers:
                self._draw_marker(shapes, m)
            if self.selected:
                sx, sy = self.w2s(self.selected.cx, self.selected.cz)
                shapes.append(cv.Circle(sx, sy, 11, ft.Paint(
                    color="#FFFFFF", stroke_width=2, style=ft.PaintingStyle.STROKE)))
            self.canvas.shapes = shapes
            self.canvas.update()
        except Exception:
            log.exception("redraw failed")

    def _marker_color(self, m):
        return m.get("color") or MARKER_STYLE.get(m["kind"], "#FFFFFF")

    def _draw_marker(self, shapes, m):
        sx, sy = self.w2s(m["x"], m["z"])
        if not (-20 <= sx <= self.canvas_w + 20 and -20 <= sy <= self.canvas_h + 20):
            return
        col = self._marker_color(m)
        # ring + crosshair so markers read clearly over ore dots
        shapes.append(cv.Circle(sx, sy, 9, ft.Paint(
            color=col, stroke_width=2.5, style=ft.PaintingStyle.STROKE)))
        shapes.append(cv.Circle(sx, sy, 2.5, ft.Paint(
            color=col, style=ft.PaintingStyle.FILL)))
        try:
            label = m.get("label") or m["kind"].title()
            shapes.append(cv.Text(sx + 11, sy - 8, label,
                          ft.TextStyle(color=col, size=12,
                                       weight=ft.FontWeight.W_600)))
        except Exception:
            pass

    def _zoom(self, factor):
        # button zoom: keep the map centre fixed
        self._zoom_at(factor, self.canvas_w / 2, self.canvas_h / 2)

    def _zoom_at(self, factor, sx, sy):
        # keep the world point currently under (sx, sy) fixed while scaling
        wpx, wpz = self.s2w(sx, sy)
        self.scale = max(0.0008, min(40.0, self.scale * factor))
        self.view_cx = wpx - (sx - self.canvas_w / 2) / self.scale
        self.view_cz = wpz - (sy - self.canvas_h / 2) / self.scale
        self._request_redraw()

    # coalesce rapid pan/zoom so the UI never backs up under a burst of events
    def _request_redraw(self):
        if self._redraw_pending:
            return
        self._redraw_pending = True
        self.page.run_task(self._redraw_soon)

    async def _redraw_soon(self):
        await asyncio.sleep(0.012)
        self._redraw_pending = False
        self._redraw()

    def _interact(self):
        """Called on pan/zoom: redraw cheaply (no terrain) and snap the terrain
        back shortly after the user stops moving."""
        self._interacting = True
        self._last_interact = time.monotonic()
        self._request_redraw()
        if not self._idle_scheduled:
            self._idle_scheduled = True
            self.page.run_task(self._idle_check)

    async def _idle_check(self):
        while time.monotonic() - self._last_interact < 0.13:
            await asyncio.sleep(0.06)
        self._idle_scheduled = False
        self._interacting = False
        self._redraw()       # final redraw, terrain included

    def _on_canvas_resize(self, e):
        self.canvas_w, self.canvas_h = e.width, e.height
        if not self._did_fit and (self.filtered or
                                  (self.cache and self.cache.get("overlay"))):
            self._fit()
            self._did_fit = True
        self._redraw()

    def _fit(self):
        # fit to the selected veins; if none are shown (e.g. on startup with all
        # materials off), frame the whole explored terrain instead
        if self.filtered:
            xs = [c.cx for c in self.filtered]
            zs = [c.cz for c in self.filtered]
            minx, maxx, minz, maxz = min(xs), max(xs), min(zs), max(zs)
        else:
            ov = self.cache.get("overlay") if self.cache else None
            if not ov:
                return
            minx, minz = ov["x0"], ov["z0"]
            maxx, maxz = ov["x0"] + ov["w"], ov["z0"] + ov["h"]
        self.view_cx = (minx + maxx) / 2
        self.view_cz = (minz + maxz) / 2
        spanx = max(50, maxx - minx)
        spanz = max(50, maxz - minz)
        self.scale = min(self.canvas_w / spanx, self.canvas_h / spanz) * 0.95

    # -------------------------------------------------------------- events
    def _on_scroll(self, e):
        dy = e.scroll_delta.y if e.scroll_delta else 0
        factor = 1.2 if dy < 0 else 1 / 1.2
        pos = getattr(e, "local_position", None)
        if pos is not None:
            self._zoom_at(factor, pos.x, pos.y)
        else:
            self._zoom(factor)

    def _on_pan(self, e):
        self.view_cx -= e.local_delta.x / self.scale
        self.view_cz -= e.local_delta.y / self.scale
        self._request_redraw()

    def _on_center(self, e):
        try:
            ex = int(float(self.center_x_f.value))
            ez = int(float(self.center_z_f.value))
        except (ValueError, TypeError):
            return
        wx, wz = ex + self.origin_x, ez + self.origin_z
        # clamp to the explored map so you never land in empty void
        bounds = None
        ov = self.cache.get("overlay") if self.cache else None
        if ov:
            bounds = (ov["x0"], ov["z0"], ov["x0"] + ov["w"], ov["z0"] + ov["h"])
        elif self._ax.shape[0]:
            bounds = (float(self._ax.min()), float(self._az.min()),
                      float(self._ax.max()), float(self._az.max()))
        off = False
        if bounds:
            cwx = min(max(wx, bounds[0]), bounds[2])
            cwz = min(max(wz, bounds[1]), bounds[3])
            off = (cwx != wx or cwz != wz)
            wx, wz = cwx, cwz
        self.view_cx, self.view_cz = wx, wz
        self.scale = max(self.scale, 0.4)        # zoom in to see the local area
        self._redraw()
        if off:
            self._set_status("That point is outside your explored map — "
                             "moved to the nearest edge.")
        else:
            self._set_status(f"Centered on ({ex}, {ez}).")

    def _on_hover(self, e):
        n = self._ax.shape[0]
        if not n:
            return
        wx, wz = self.s2w(e.local_position.x, e.local_position.y)
        d2 = (self._ax - wx) ** 2 + (self._az - wz) ** 2
        i = int(np.argmin(d2))
        # within ~12 screen px?
        if (d2[i] ** 0.5) * self.scale > 12:
            if self._hover_idx != -1:
                self._hover_idx = -1
                self.hover_label.value = "Hover a vein to see its coordinates"
                self.hover_label.color = "#8A94A6"
                self.hover_label.update()
            return
        if i == self._hover_idx:
            return
        self._hover_idx = i
        c = self.draw_order[i]
        col = METAL_COLORS.get(c.metal, "#FFFFFF")
        dx, dz = c.cx - self.origin_x, c.cz - self.origin_z
        self.hover_label.value = (f"{c.mineral} {c.best_grade or ''} ×{c.count}  "
                                  f"@ ({dx}, {c.cy}, {dz})  y{c.ymin}-{c.ymax}")
        self.hover_label.color = col
        self.hover_label.update()

    def _on_tap(self, e):
        wx, wz = self.s2w(e.local_position.x, e.local_position.y)
        if self.place_mode:
            self._add_marker(int(round(wx)), int(round(wz)))
            return
        best, bd = None, 1e18
        for c in self.filtered:
            d = (c.cx - wx) ** 2 + (c.cz - wz) ** 2
            if d < bd:
                bd, best = d, c
        if best and bd ** 0.5 * self.scale < 14:
            self._select(best)

    def _on_overlay_toggle(self, e):
        self.show_overlay = e.control.value
        self._redraw()

    def _make_density_layer(self, name, label, data_key, types, all_lbl, build_fn):
        """Register one density overlay (trees/rocks/gravel/sand). All four are
        identical apart from their data + colour ramp, so they share handlers."""
        sw = ft.Switch(
            value=False, scale=0.7,
            on_change=lambda e, n=name: self._on_density_toggle(n, e.control.value))
        dd = ft.Dropdown(
            value=all_lbl, width=150, dense=True,
            on_select=lambda e, n=name: self._on_density_type(n),
            options=[ft.dropdown.Option(all_lbl)] +
                    [ft.dropdown.Option(t.capitalize()) for t in sorted(types)])
        self.density_layers[name] = dict(
            label=label, data_key=data_key, types=list(types), all_lbl=all_lbl,
            build_fn=build_fn, mode=False, idx=None, img=None, url=None,
            bounds=None, seq=0, prev=None, switch=sw, dropdown=dd)

    def _on_density_toggle(self, name, value):
        L = self.density_layers[name]
        L["mode"] = value
        if value:                       # only one density layer on at a time
            for n2, L2 in self.density_layers.items():
                if n2 != name:
                    L2["mode"] = False
                    L2["switch"].value = False
            self._regen_layer(name)
        self._redraw()

    def _on_density_type(self, name):
        L = self.density_layers[name]
        v = L["dropdown"].value
        L["idx"] = None if v == L["all_lbl"] else L["types"].index(v.lower())
        if L["mode"]:
            self._regen_layer(name)
        self._redraw()

    def _regen_layer(self, name):
        L = self.density_layers[name]
        if not (self.cache and self.cache.get(L["data_key"])):
            L["url"] = None
            self._set_status(f"This world hasn't been re-scanned for "
                             f"{L['label'].lower()} yet — click Rescan.", err=True)
            return
        try:
            assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
            os.makedirs(assets, exist_ok=True)
            L["seq"] += 1
            fn = f"{name}_{L['seq']}.png"
            info = L["build_fn"](self.cache, L["idx"], os.path.join(assets, fn))
            if not info:
                L["url"] = None
                self._set_status(f"No {L['label'].lower()} of that kind in the "
                                 "explored area.")
                return
            if L["prev"]:
                try:
                    os.remove(os.path.join(assets, L["prev"]))
                except OSError:
                    pass
            L["prev"] = fn
            L["url"] = f"/{fn}"
            L["bounds"] = (info["x0"], info["z0"], info["w"], info["h"])
            L["img"] = cv.Image(src=L["url"])
        except Exception:
            log.exception("density overlay %s failed", name)
            L["url"] = None

    def _on_filter(self, e):
        self.enabled_metals = {m for m, sw in self.metal_switches.items() if sw.value}
        self._apply_filter()
        # if you just enabled materials but none are in view, jump to them
        if self.filtered and not self._any_visible():
            self._fit()
        self._redraw()

    def _any_visible(self):
        if self._ax.shape[0] == 0:
            return False
        sx = (self._ax - self.view_cx) * self.scale + self.canvas_w / 2
        sy = (self._az - self.view_cz) * self.scale + self.canvas_h / 2
        return bool(np.any((sx >= 0) & (sx <= self.canvas_w)
                           & (sy >= 0) & (sy <= self.canvas_h)))

    # ------------------------------------------------------------- markers
    def _markers_path(self, save=None):
        save = save or self.save_dd.value
        return scanner.cache_path(save).replace(".orecache", ".markers.json")

    def _load_markers(self):
        self.markers = []
        try:
            p = self._markers_path()
            if os.path.exists(p):
                with open(p, "r", encoding="utf8") as f:
                    self.markers = json.load(f)
        except Exception:
            log.exception("load markers failed")

    def _save_markers(self):
        try:
            p = self._markers_path()
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf8") as f:
                json.dump(self.markers, f)
        except Exception:
            log.exception("save markers failed")

    def _toggle_place(self, e):
        self.place_mode = not self.place_mode
        self.place_btn.text = "Click map to place ✓" if self.place_mode else "Place marker on map"
        self.place_btn.style = ft.ButtonStyle(
            bgcolor="#2A6B4F" if self.place_mode else None)
        self.page.update()

    def _add_marker(self, x, z):
        kind = MARKER_KINDS.get(self.marker_kind_dd.value, "note")
        label = (self.marker_label_f.value or "").strip()
        self.markers.append({"x": x, "z": z, "kind": kind, "label": label})
        self.marker_label_f.value = ""
        self._disarm_clear()
        self._save_markers()
        self._refresh_markers()
        self._redraw()

    def _del_marker(self, m):
        try:
            self.markers.remove(m)
        except ValueError:
            return
        if m is self._editing:
            self._editing = None
        if m is self._coloring:
            self._coloring = None
        self._disarm_clear()
        self._save_markers()
        self._refresh_markers()
        self._redraw()

    def _disarm_clear(self):
        """Reset the two-step 'Remove markers' confirm back to its idle state."""
        if self._clear_armed:
            self._clear_armed = False
            self.clear_markers_btn.text = "Remove markers"
            self.clear_markers_btn.style = None

    def _on_clear_markers(self, e):
        n = sum(1 for m in self.markers if m.get("kind") != "base")
        if n == 0:
            self._disarm_clear()
            self._set_status("No non-base markers to remove — Base markers are kept.")
            self.page.update()
            return
        if not self._clear_armed:          # first click: arm + ask to confirm
            self._clear_armed = True
            self.clear_markers_btn.text = f"Confirm — remove {n}?"
            self.clear_markers_btn.style = ft.ButtonStyle(color="#FF6B6B")
            self.page.update()
            return
        # second click: do it (keep only base markers)
        self.markers = [m for m in self.markers if m.get("kind") == "base"]
        self._editing = None
        self._coloring = None
        self._disarm_clear()
        self._save_markers()
        self._refresh_markers()
        self._redraw()
        self._set_status(f"Removed {n} marker(s); kept Base markers.")

    def _edit_marker(self, m):
        self._editing = m
        self._coloring = None
        self._disarm_clear()
        self._refresh_markers()

    def _save_marker_label(self, m, value):
        m["label"] = (value or "").strip()
        self._editing = None
        self._save_markers()
        self._refresh_markers()
        self._redraw()

    def _cancel_edit(self, *_):
        self._editing = None
        self._refresh_markers()

    def _open_color(self, m):
        # toggle the inline colour picker for this marker
        self._coloring = None if self._coloring is m else m
        self._editing = None
        self._disarm_clear()
        self._refresh_markers()

    def _set_marker_color(self, m, color):
        if color is None:
            m.pop("color", None)       # revert to the type's default colour
        else:
            m["color"] = color
        self._coloring = None
        self._save_markers()
        self._refresh_markers()
        self._redraw()

    def _marker_origin(self, m):
        self.origin_x, self.origin_z = m["x"], m["z"]
        self.origin_x_f.value = str(m["x"])
        self.origin_z_f.value = str(m["z"])
        self._refresh_list()
        if self.selected:
            self._select(self.selected)

    def _refresh_markers(self):
        rows = []
        for m in self.markers:
            col = self._marker_color(m)
            dx, dz = m["x"] - self.origin_x, m["z"] - self.origin_z
            # the colour dot doubles as a button that opens the colour picker
            dot = ft.Container(width=13, height=13, bgcolor=col, border_radius=7,
                               border=ft.Border.all(1, "#0B0E14"),
                               tooltip="Change colour",
                               on_click=lambda e, m=m: self._open_color(m))
            if m is self._editing:
                tf = ft.TextField(
                    value=m.get("label", ""), dense=True, expand=True, autofocus=True,
                    hint_text=m["kind"].title(),
                    on_submit=lambda e, m=m: self._save_marker_label(m, e.control.value))
                rows.append(ft.Row([
                    dot, tf,
                    ft.IconButton(ft.Icons.CHECK, icon_size=15, tooltip="Save name",
                                  on_click=lambda e, m=m, tf=tf:
                                  self._save_marker_label(m, tf.value)),
                    ft.IconButton(ft.Icons.CLOSE, icon_size=15, tooltip="Cancel",
                                  on_click=self._cancel_edit)],
                    spacing=2, tight=True))
            else:
                rows.append(ft.Row([
                    dot,
                    ft.Column([
                        ft.Text(m.get("label") or m["kind"].title(), size=11,
                                weight=ft.FontWeight.W_600),
                        ft.Text(f"({dx}, {dz})", size=10, color="#8A94A6")],
                        spacing=0, expand=True),
                    ft.IconButton(ft.Icons.EDIT_OUTLINED, icon_size=14, tooltip="Rename",
                                  on_click=lambda e, m=m: self._edit_marker(m)),
                    ft.IconButton(ft.Icons.MY_LOCATION, icon_size=14, tooltip="Use as origin",
                                  on_click=lambda e, m=m: self._marker_origin(m)),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, icon_size=14, tooltip="Delete",
                                  on_click=lambda e, m=m: self._del_marker(m))],
                    spacing=0, tight=True))
            if m is self._coloring:
                rows.append(self._color_palette_row(m))
        self.markers_list.controls = rows
        self.page.update()

    def _color_palette_row(self, m):
        def swatch(c):
            selected = (m.get("color") == c)
            return ft.Container(
                width=19, height=19, bgcolor=c, border_radius=10, tooltip=c,
                border=ft.Border.all(2, "#FFFFFF" if selected else "#0B0E14"),
                on_click=lambda e, c=c, m=m: self._set_marker_color(m, c))
        chips = [swatch(c) for c in MARKER_PALETTE]
        chips.append(ft.TextButton("Default", icon=ft.Icons.RESTART_ALT,
                                   tooltip="Revert to the marker type's colour",
                                   on_click=lambda e, m=m: self._set_marker_color(m, None)))
        return ft.Container(
            ft.Row(chips, wrap=True, spacing=5, run_spacing=5),
            padding=ft.Padding.only(left=16, top=2, bottom=6))

    def _set_all_metals(self, val):
        for sw in self.metal_switches.values():
            sw.value = val
        self.page.update()
        self._on_filter(None)

    def _toggle_group(self, items):
        # turn the whole category on if any are off, else turn all off
        keys = [k for k, _, _ in items]
        target = not all(self.metal_switches[k].value for k in keys)
        for k in keys:
            self.metal_switches[k].value = target
        self.page.update()
        self._on_filter(None)

    def _on_origin(self, e):
        try:
            self.origin_x = int(float(self.origin_x_f.value or 0))
            self.origin_z = int(float(self.origin_z_f.value or 0))
        except ValueError:
            return
        self._refresh_list()
        if self.selected:
            self._select(self.selected)

    def _on_merge_change(self, e):
        if not self.cache or self.busy:
            return
        self.page.run_task(self._recluster)

    async def _recluster(self):
        self.busy = True
        self._set_status("Re-merging deposits…")
        size = int(self.cluster_slider.value)
        clusters = await asyncio.to_thread(cluster_from_cache, self.cache, size)
        self.clusters = clusters
        if self.cache is not None:
            self.cache["cluster_size"] = size
        self._apply_filter()
        self._redraw()
        self._set_status(f"Merged at radius {size}: {len(self.clusters):,} deposits.")
        self.busy = False

    # -------------------------------------------------------------- filter
    def _apply_filter(self):
        self.filtered = [c for c in self.clusters if c.metal in self.enabled_metals]
        # Build a metal-fair draw order: round-robin across metals so no single
        # metal (e.g. gold) consumes the whole draw budget. Each metal's deposits
        # stay value-sorted internally (self.clusters is pre-sorted by priority).
        from collections import defaultdict, deque
        groups: dict = defaultdict(deque)
        for c in self.filtered:
            groups[c.metal].append(c)
        queues = list(groups.values())
        order = []
        while queues:
            nxt = []
            for q in queues:
                order.append(q.popleft())
                if q:
                    nxt.append(q)
            queues = nxt
        self.draw_order = order
        # precompute numpy arrays for fast vectorised drawing
        n = len(order)
        self._ax = np.fromiter((c.cx for c in order), np.float64, n) if n else np.zeros(0)
        self._az = np.fromiter((c.cz for c in order), np.float64, n) if n else np.zeros(0)
        # dot size reflects how many ore blocks are in the vein, so the merge
        # radius is visible: bigger merges -> fewer, larger dots
        self._acount = (np.fromiter((c.count for c in order), np.int64, n)
                        if n else np.zeros(0, np.int64))
        self._asize = (np.clip(3 + np.sqrt(self._acount.astype(float)) * 1.3, 4, 18)
                       .round().astype(int) if n else np.zeros(0, int))
        self._acol = [METAL_COLORS.get(c.metal, "#FFFFFF") for c in order]
        self._refresh_list()

    def _grouped_for_list(self, cell=100):
        """Combine nearby deposits of the same material into one list entry, so
        the list is short and easy to skim. (The map dots are unaffected.)"""
        agg: dict = {}
        for c in self.filtered:
            key = (c.metal, int(c.cx) // cell, int(c.cz) // cell)
            g = agg.get(key)
            if g is None:
                agg[key] = g = dict(metal=c.metal, mineral=c.mineral, sx=0, sy=0,
                                    sz=0, n=0, ymin=c.ymin, ymax=c.ymax,
                                    grade=c.best_grade, veins=0, samples=[])
            g["sx"] += c.cx * c.count
            g["sy"] += c.cy * c.count
            g["sz"] += c.cz * c.count
            g["n"] += c.count
            g["ymin"] = min(g["ymin"], c.ymin)
            g["ymax"] = max(g["ymax"], c.ymax)
            g["veins"] += 1
            if GRADE_RANK.get(c.best_grade, 0) > GRADE_RANK.get(g["grade"], 0):
                g["grade"] = c.best_grade
            if len(g["samples"]) < 5 and c.samples:
                g["samples"].append(c.samples[0])
        out = []
        for g in agg.values():
            n = max(1, g["n"])
            oc = OreCluster(metal=g["metal"], mineral=g["mineral"],
                            best_grade=g["grade"], count=g["n"],
                            cx=g["sx"] // n, cy=g["sy"] // n, cz=g["sz"] // n,
                            ymin=g["ymin"], ymax=g["ymax"], samples=g["samples"])
            oc.veins = g["veins"]
            out.append(oc)
        out.sort(key=lambda c: c.count, reverse=True)
        return out

    def _base_point(self):
        """Reference point for distance sorting: the Base marker if placed, else
        any marker, else the coordinate origin (spawn)."""
        for m in self.markers:
            if m["kind"] == "base":
                return m["x"], m["z"]
        if self.markers:
            return self.markers[0]["x"], self.markers[0]["z"]
        return self.origin_x, self.origin_z

    def _on_sort(self, e):
        self.list_sort = {"Biggest": "biggest", "Nearest base": "nearest",
                          "Best value (near + rich)": "value"}.get(
                              self.sort_dd.value, "biggest")
        self._refresh_list()

    def _refresh_list(self):
        try:
            ranked = self._grouped_for_list()
            bx, bz = self._base_point()
            for c in ranked:
                c.dist = ((c.cx - bx) ** 2 + (c.cz - bz) ** 2) ** 0.5
            if self.list_sort == "nearest":
                ranked.sort(key=lambda c: c.dist)
            elif self.list_sort == "value":
                # reward rich + close: more blocks, higher grade, shorter travel
                ranked.sort(key=lambda c: -(c.count * (1 + GRADE_RANK.get(
                    c.best_grade, 0)) / (c.dist + 300)))
            else:
                ranked.sort(key=lambda c: -c.count)
            self.list_view.controls = [self._list_tile(c) for c in ranked[:250]]
            self.stats_txt.value = (
                f"{len(ranked):,} areas · {len(self.filtered):,} veins\n"
                f"{sum(c.count for c in self.filtered):,} ore blocks")
            self.merge_count_txt.value = (
                f"≈ {len(self.clusters):,} veins at {int(self.cluster_slider.value)}-block "
                f"grouping")
            self.page.update()
        except Exception:
            log.exception("refresh_list failed")

    def _list_tile(self, c):
        col = METAL_COLORS.get(c.metal, "#FFFFFF")
        dx, dz = c.cx - self.origin_x, c.cz - self.origin_z
        veins = getattr(c, "veins", 1)
        vtxt = f"  ·  {veins} veins" if veins > 1 else ""
        d = getattr(c, "dist", None)
        dtxt = ""
        if d is not None:
            dtxt = (f"  ·  {int(d):,} away" if d < 1000
                    else f"  ·  {d/1000:.1f}k away")
        return ft.Container(
            padding=8, border_radius=8, bgcolor="#1A1F2B",
            on_click=lambda e, c=c: self._select(c, center=True),
            content=ft.Row([
                ft.Container(width=10, height=10, bgcolor=col, border_radius=5),
                ft.Column([
                    ft.Text(f"{c.mineral}  ·  {c.best_grade or 'ore'}{dtxt}",
                            size=12, weight=ft.FontWeight.W_600),
                    ft.Text(f"({dx}, {c.cy}, {dz})   ·  ×{c.count}{vtxt}  ·  "
                            f"y{c.ymin}-{c.ymax}",
                            size=11, color="#8A94A6")],
                    spacing=1, expand=True)], spacing=8))

    def _select(self, c, center=False):
        self.selected = c
        if center:                       # clicked from the list -> fly to it
            self.view_cx, self.view_cz = c.cx, c.cz
            if self.scale < 2.0:
                self.scale = 2.0
        col = METAL_COLORS.get(c.metal, "#FFFFFF")
        dx, dz = c.cx - self.origin_x, c.cz - self.origin_z
        samples = "\n".join(f"   • ({x-self.origin_x}, {y}, {z-self.origin_z})"
                            for x, y, z in c.samples)
        self.detail.content = ft.Column([
            ft.Row([ft.Container(width=14, height=14, bgcolor=col, border_radius=7),
                    ft.Text(c.mineral, size=15, weight=ft.FontWeight.BOLD)], spacing=8),
            ft.Text(f"{c.best_grade or 'ore'} grade · {c.count} blocks", size=12,
                    color="#8A94A6"),
            ft.Container(height=6),
            ft.Text(f"Centre:  {dx},  {c.cy},  {dz}", size=14,
                    weight=ft.FontWeight.W_600, selectable=True),
            ft.Text(f"Depth (Y): {c.ymin} to {c.ymax}", size=12, color="#8A94A6"),
            ft.Container(height=6),
            ft.Text("Exact ore blocks:", size=12, color="#8A94A6"),
            ft.Text(samples or "   (none)", size=12, selectable=True)], spacing=3)
        self._redraw()
        self.page.update()

    # --------------------------------------------------------------- scan
    def _set_status(self, msg, err=False):
        self.status.value = msg
        self.status.color = "#E36A4B" if err else "#8A94A6"
        self.page.update()

    # ----------------------------------------------------- window controls
    def _win_min(self, e):
        self.page.window.minimized = True
        self.page.update()

    def _win_max(self, e):
        self.page.window.maximized = not self.page.window.maximized
        self.page.update()

    def _win_close(self, e):
        self.page.window.close()

    def _on_refresh(self, e):
        if self.busy:
            return
        if not self.save_dd.value:
            self._set_status("No save selected.", err=True)
            return
        self.busy = True
        self.refresh_btn.disabled = True
        self.progress.visible = True
        self.progress.value = None
        self.page.run_task(self._scan_async)

    def _on_autorescan_toggle(self, e):
        self.auto_rescan = e.control.value
        if self.auto_rescan and not self._auto_running:
            self.page.run_task(self._auto_rescan_loop)
        mins = self.auto_interval_sec // 60
        self._set_status(f"Auto-rescan on — updates every {mins} min."
                         if self.auto_rescan else "Auto-rescan off.")

    def _on_auto_interval(self, e):
        try:
            self.auto_interval_sec = int(self.auto_interval_dd.value) * 60
        except (ValueError, TypeError):
            pass

    async def _auto_rescan_loop(self):
        self._auto_running = True
        try:
            while self.auto_rescan:
                for _ in range(self.auto_interval_sec):   # 1s steps so toggle-off is prompt
                    if not self.auto_rescan:
                        break
                    await asyncio.sleep(1)
                if not self.auto_rescan or self.busy or not self.save_dd.value:
                    continue
                self.busy = True
                self.refresh_btn.disabled = True
                self.progress.visible = True
                self.progress.value = None
                await self._scan_async(refit=False)   # keep the user's view
        finally:
            self._auto_running = False

    async def _scan_async(self, refit=True):
        save = self.save_dd.value
        size = int(self.cluster_slider.value)
        state = {"d": 0, "t": 1, "m": ""}

        def prog(done, total, msg):
            state.update(d=done, t=total, m=msg)

        def blocking():
            if self.scanner is None:
                self.scanner = Scanner(game_dir=self.game_dir)
            return self.scanner.scan_to_cache(
                save, cluster_size=size, incremental=True, progress=prog)

        task = asyncio.create_task(asyncio.to_thread(blocking))
        try:
            while not task.done():
                d, t = state["d"], state["t"]
                self.progress.value = (d / t) if t else None
                self.status.value = f"Scanning… {d:,}/{t:,} chunks  ({state['m']})"
                self.page.update()
                await asyncio.sleep(0.25)
            cache = task.result()
            self.cache = cache
            self.clusters = cache.get("clusters", [])
            self._apply_filter()
            if refit:
                self._fit()
            self._redraw()
            st = cache["stats"]
            msg = (f"✓ {st['clusters']:,} deposits · {st['decoded']:,} chunks re-read, "
                   f"{st['reused']:,} unchanged · {st['ore_blocks']:,} ore blocks")
            mined = cache.get("mined_delta") or {}
            if mined:
                top = sorted(mined.items(), key=lambda kv: -kv[1])[:4]
                parts = [f"{MATERIAL_LABEL.get(m, m.title())} −{n:,}" for m, n in top]
                msg += "  ·  ⛏ mined out: " + ", ".join(parts)
            self._set_status(msg)
        except Exception as ex:
            log.exception("scan failed")
            traceback.print_exc()
            if "install not found" in str(ex):
                # game DLLs not where we expected — let the user point us at them
                self._set_status("Vintage Story install not found — opening a "
                                 "folder picker to locate it…", err=True)
                await self._on_browse_gamedir()
            else:
                self._set_status(f"Error: {ex}  (see orefinder.log)", err=True)
        finally:
            self.busy = False
            self.refresh_btn.disabled = False
            self.progress.visible = False
            self.page.update()


def main(page: ft.Page):
    page.window.width = 1600
    page.window.height = 920
    page.window.title_bar_hidden = True          # use our custom dark title bar
    page.window.title_bar_buttons_hidden = True
    ico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
    if os.path.exists(ico):
        page.window.icon = ico
    try:
        OreFinderApp(page)
    except Exception:
        log.exception("startup failed")
        raise


if __name__ == "__main__":
    ft.run(main, assets_dir="assets")
