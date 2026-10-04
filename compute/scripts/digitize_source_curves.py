# -*- coding: utf-8 -*-
"""Digitise the five measured triaxial curves and the three pore-pressure
histories from the printed panels of the source paper.

The panels are 300-dpi rasters embedded in the source PDF, so this script reads
those rasters at their native resolution rather than re-rendering the page.
Axis calibration comes from the plotted panel frame, checked against the printed
tick marks before any curve is read.

Reading rule. A column scan maps one axial-strain column to one ordinate, which
is well defined only while the plotted record is single-valued in axial strain.
On three of the five deviator panels it is not: near the end of the test the
record doubles back, so the loading branch and the branch the specimen follows
after the reversal occupy the same strain columns. Because the ordinate can only
fall across a reversal, the loading branch is the upper of the two, and this
script reads it: outside a reversal interval the ordinate is the median of the
red band in the column; inside one it is the median of the top of the uppermost
red band, over a depth equal to the panel's own plotted line thickness. The
reversal intervals are detected from the panel, not assumed, and are reported in
the output.

Usage:
    python scripts/digitize_source_curves.py [--pdf PATH] [--stations 70]
        [--out-dir results/data/nordas_curves] [--report-only]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

# --- source ---------------------------------------------------------------

DEFAULT_PDF = r"I:\GeoSkills\databases\pipeline_workspace\pdfs\ct_180.pdf"
SOURCE_PAPER = "R02_Nordas_2024"

# Native embedded raster per source page (0-based page index, image xref).
PAGE_RASTER = {6: (5, 126), 7: (6, 142)}

# Red trace mask, on the native raster.
RED = dict(r_min=150, g_max=90, b_max=90, rg_min=80, rb_min=80)

# Two red bands in one column belong to one band interrupted by an overplotted
# model line when either test holds: the separation is no wider than the widest
# interruption seen on the two panels whose record never reverses (CU-S, CU-Z),
# or the separation is itself filled with plotted ink, which is what a model line
# crossing a steep part of the trace leaves behind. A separation that is both
# wider than that bound and mostly blank page is a real branch separation.
MERGE_PX = 16
GAP_INK_FRACTION = 0.6

# A reversal interval must be a sustained block of multi-valued columns; short
# blocks are line crossings that survived the merge.
MIN_REVERSAL_COLS = 8

# Panel frames: (left, right, top, bottom) spine positions in native raster
# pixels, verified against the raster by check_frame().
PANELS = [
    dict(key="fig2a_CU_S", page=6, box=(126.5, 643.0, 485.5, 895.5),
         ylim=(60.0, 0.0), quantity="q", specimen_id="A9_TRU1_1",
         test_type="CU", theta_deg=90.0, sigma_r0_MPa=13.0,
         source_locator="Nordas 2024 Fig.2a (page 6)"),
    dict(key="fig2b_CU_P", page=6, box=(807.5, 1325.0, 485.5, 895.5),
         ylim=(60.0, 0.0), quantity="q", specimen_id="B8_TRU1_1",
         test_type="CU", theta_deg=0.0, sigma_r0_MPa=13.0,
         source_locator="Nordas 2024 Fig.2b (page 6)"),
    dict(key="fig2c_CU_Z", page=6, box=(1491.5, 2007.5, 485.5, 895.5),
         ylim=(60.0, 0.0), quantity="q", specimen_id="C13T_BUL1-1",
         test_type="CU", theta_deg=30.0, sigma_r0_MPa=13.0,
         source_locator="Nordas 2024 Fig.2c (page 6)"),
    dict(key="fig3a_CD_S", page=7, box=(125.0, 636.5, 477.5, 890.5),
         ylim=(40.0, 0.0), quantity="q", specimen_id="B2_MAR1_1",
         test_type="CD", theta_deg=90.0, sigma_r0_MPa=5.0,
         source_locator="Nordas 2024 Fig.3a (page 7)"),
    dict(key="fig3b_CD_P", page=7, box=(840.5, 1353.0, 477.5, 890.5),
         ylim=(40.0, 0.0), quantity="q", specimen_id="A12_TRU1_1",
         test_type="CD", theta_deg=0.0, sigma_r0_MPa=5.0,
         source_locator="Nordas 2024 Fig.3b (page 7)"),
    dict(key="fig2d_CU_S_pw", page=6, box=(126.5, 643.0, 1075.5, 1489.5),
         ylim=(12.0, -4.0), quantity="pw", specimen_id="A9_TRU1_1",
         test_type="CU", theta_deg=90.0, sigma_r0_eff_MPa=13.0,
         source_locator="Nordas 2024 Fig.2d (page 6)"),
    dict(key="fig2e_CU_P_pw", page=6, box=(807.5, 1325.0, 1075.5, 1489.5),
         ylim=(12.0, -4.0), quantity="pw", specimen_id="B8_TRU1_1",
         test_type="CU", theta_deg=0.0, sigma_r0_eff_MPa=13.0,
         source_locator="Nordas 2024 Fig.2e (page 6)"),
    dict(key="fig2f_CU_Z_pw", page=6, box=(1491.5, 2007.5, 1075.5, 1489.5),
         ylim=(12.0, -4.0), quantity="pw", specimen_id="C13T_BUL1-1",
         test_type="CU", theta_deg=30.0, sigma_r0_eff_MPa=13.0,
         source_locator="Nordas 2024 Fig.2f (page 6)"),
]

XLIM = (0.0, 1.5)          # axial strain, per cent, every panel
XTICKS = (0.0, 0.5, 1.0, 1.5)

# The deviator panel whose reversal interval also governs its specimen's
# pore-pressure panel: same specimen, same axial-strain abscissa.
PW_OF = {"fig2d_CU_S_pw": "fig2a_CU_S",
         "fig2e_CU_P_pw": "fig2b_CU_P",
         "fig2f_CU_Z_pw": "fig2c_CU_Z"}


# --- raster ---------------------------------------------------------------

def load_rasters(pdf_path: Path) -> dict[int, np.ndarray]:
    doc = fitz.open(pdf_path)
    out = {}
    for page_no, (pidx, xref) in PAGE_RASTER.items():
        placed = doc[pidx].get_image_rects(xref)
        if not placed:
            raise SystemExit(f"page {page_no}: image xref {xref} is not placed")
        info = doc.extract_image(xref)
        out[page_no] = np.asarray(Image.open(io.BytesIO(info["image"])).convert("RGB"))
    return out


def red_mask(a: np.ndarray) -> np.ndarray:
    r = a[:, :, 0].astype(int)
    g = a[:, :, 1].astype(int)
    b = a[:, :, 2].astype(int)
    return ((r > RED["r_min"]) & (g < RED["g_max"]) & (b < RED["b_max"])
            & (r - g > RED["rg_min"]) & (r - b > RED["rb_min"]))


def dark_mask(a: np.ndarray) -> np.ndarray:
    return (a[:, :, 0] < 110) & (a[:, :, 1] < 110) & (a[:, :, 2] < 110)


def ink_mask(a: np.ndarray) -> np.ndarray:
    """Anything drawn on the panel, of any colour, as against the blank page."""
    return ~((a[:, :, 0] > 230) & (a[:, :, 1] > 230) & (a[:, :, 2] > 230))


def check_frame(dark: np.ndarray, box, ylim, key: str) -> dict:
    """The four spines must really be drawn, and the x ticks must fall where the
    frame calibration puts them."""
    x0, x1, ytop, ybot = box
    xi0, xi1 = int(round(x0)), int(round(x1))
    yi0, yi1 = int(round(ytop)), int(round(ybot))

    def line_frac(sl_y, sl_x):
        seg = dark[sl_y, sl_x]
        return float(seg.max(axis=0).mean() if seg.ndim == 2 else seg.mean())

    fr = dict(
        left=line_frac(slice(yi0, yi1 + 1), slice(xi0 - 1, xi0 + 2)),
        right=line_frac(slice(yi0, yi1 + 1), slice(xi1 - 1, xi1 + 2)),
        top=line_frac(slice(yi0 - 1, yi0 + 2), slice(xi0, xi1 + 1)),
        bottom=line_frac(slice(yi1 - 1, yi1 + 2), slice(xi0, xi1 + 1)),
    )
    # transpose handling: for top/bottom the reduction axis is rows
    fr["top"] = float(dark[yi0 - 1:yi0 + 2, xi0:xi1 + 1].max(axis=0).mean())
    fr["bottom"] = float(dark[yi1 - 1:yi1 + 2, xi0:xi1 + 1].max(axis=0).mean())
    fr["left"] = float(dark[yi0:yi1 + 1, xi0 - 1:xi0 + 2].max(axis=1).mean())
    fr["right"] = float(dark[yi0:yi1 + 1, xi1 - 1:xi1 + 2].max(axis=1).mean())
    for side, v in fr.items():
        if v < 0.90:
            raise SystemExit(f"{key}: {side} spine is only {v:.2f} dark — frame box is wrong")

    # Tick marks hang below the abscissa. Where the ordinate range spans zero the
    # abscissa is drawn at zero rather than at the bottom spine.
    yhi, ylo = ylim
    if ylo < 0.0 < yhi:
        base = int(round(ytop + (0.0 - yhi) / (ylo - yhi) * (ybot - ytop)))
    else:
        base = yi1
    ticks = {}
    for t in XTICKS:
        px = x0 + (t - XLIM[0]) / (XLIM[1] - XLIM[0]) * (x1 - x0)
        j = int(round(px))
        below = dark[base + 2:base + 10, max(j - 3, 0):j + 4]
        ticks[t] = bool(below.any())
    missing = [t for t, ok in ticks.items() if not ok]
    # the left and right ticks coincide with the spines and need not be redrawn
    missing = [t for t in missing if t not in (XLIM[0], XLIM[1])]
    if missing:
        raise SystemExit(f"{key}: no tick mark under x={missing} — x calibration is wrong")
    return fr


# --- column structure -----------------------------------------------------

def column_bands(col: np.ndarray, ink: np.ndarray,
                 merge: int = MERGE_PX) -> list[tuple[int, int]]:
    """Contiguous red runs in one column, merging runs an overplotted line split."""
    runs, start = [], None
    for i, v in enumerate(col):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(col) - 1])
    if not runs:
        return []
    out = [runs[0]]
    for a, b in runs[1:]:
        lo, hi = out[-1][1] + 1, a
        gap = hi - lo
        filled = gap > 0 and float(ink[lo:hi].mean()) >= GAP_INK_FRACTION
        if gap <= merge or filled:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def reversal_blocks(bands: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Sustained runs of columns carrying more than one band."""
    multi = [j for j, bb in enumerate(bands) if len(bb) >= 2]
    if not multi:
        return []
    groups, run = [], [multi[0]]
    for j in multi[1:]:
        if j - run[-1] > 3:
            groups.append(run)
            run = []
        run.append(j)
    groups.append(run)
    return [(g[0], g[-1]) for g in groups if len(g) >= MIN_REVERSAL_COLS]


def line_thickness(bands: list[list[tuple[int, int]]]) -> int:
    """The panel's own plotted line thickness: the median height of a band in a
    column carrying exactly one, over the columns where the trace is flattest."""
    h = [bb[0][1] - bb[0][0] + 1 for bb in bands if len(bb) == 1]
    if not h:
        raise SystemExit("no single-band column to measure line thickness from")
    return int(max(3, np.median(sorted(h)[:max(len(h) // 2, 1)])))


# --- reading --------------------------------------------------------------

def read_panel(mask: np.ndarray, ink: np.ndarray, panel: dict,
               inherit_reversals=None) -> dict:
    x0, x1, ytop, ybot = panel["box"]
    yhi, ylo = panel["ylim"]
    ci0, ci1 = int(round(x0)) + 2, int(round(x1)) - 2
    ri0, ri1 = int(round(ytop)) + 2, int(round(ybot)) - 2
    sub = mask[ri0:ri1, ci0:ci1]
    sub_ink = ink[ri0:ri1, ci0:ci1]

    def to_x(j):
        return XLIM[0] + (ci0 + j - x0) / (x1 - x0) * (XLIM[1] - XLIM[0])

    def to_y(i):
        return yhi + (ri0 + i - ytop) / (ybot - ytop) * (ylo - yhi)

    bands = [column_bands(sub[:, j], sub_ink[:, j]) for j in range(sub.shape[1])]
    red_cols = [j for j, bb in enumerate(bands) if bb]
    if not red_cols:
        raise SystemExit(f"{panel['key']}: no red trace found")
    thick = line_thickness(bands)

    blocks = inherit_reversals
    if blocks is None:
        cols = reversal_blocks(bands)
        # A reversal runs from where the returning branch separates to the turning
        # point, and the turning point is where the record stops: past the block the
        # two branches have converged but still share their columns.
        if cols:
            cols[-1] = (cols[-1][0], red_cols[-1])
        blocks = [(to_x(a), to_x(b)) for a, b in cols]
    in_reversal = [any(lo <= to_x(j) <= hi for lo, hi in blocks)
                   for j in range(sub.shape[1])]

    xs, ys = [], []
    for j in red_cols:
        if in_reversal[j]:
            a, b = bands[j][0]                # uppermost band = the loading branch
            b = min(b, a + thick - 1)         # its top, one plotted line deep
            row = (a + b) / 2.0
        else:
            row = float(np.median(np.flatnonzero(sub[:, j])))
        xs.append(to_x(j))
        ys.append(to_y(row))

    return dict(x=np.array(xs), y=np.array(ys), n_red_columns=len(red_cols),
                n_panel_columns=sub.shape[1], line_thickness_px=thick,
                reversal_intervals=[(round(lo, 4), round(hi, 4)) for lo, hi in blocks],
                px_per_unit=abs((ybot - ytop) / (yhi - ylo)))


def resample(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(x.min(), x.max(), n)
    return grid, np.interp(grid, x, y)


# --- output ---------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--out-dir", default="results/data/nordas_curves")
    ap.add_argument("--stations", type=int, default=70)
    ap.add_argument("--report-only", action="store_true",
                    help="print the panel report and write nothing")
    args = ap.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"source PDF not found: {pdf}")
    rasters = load_rasters(pdf)
    masks = {p: red_mask(a) for p, a in rasters.items()}
    darks = {p: dark_mask(a) for p, a in rasters.items()}
    inks = {p: ink_mask(a) for p, a in rasters.items()}

    read: dict[str, dict] = {}
    for panel in PANELS:
        check_frame(darks[panel["page"]], panel["box"], panel["ylim"], panel["key"])
        inherit = None
        if panel["key"] in PW_OF:
            inherit = read[PW_OF[panel["key"]]]["reversal_intervals"]
        read[panel["key"]] = read_panel(masks[panel["page"]], inks[panel["page"]],
                                        panel, inherit)

    print(f"{'panel':<16} {'cols':>5} {'thick':>6} {'x range':>16}  reversal intervals")
    for panel in PANELS:
        r = read[panel["key"]]
        iv = ", ".join(f"{lo:.4f}-{hi:.4f}" for lo, hi in r["reversal_intervals"]) or "none"
        print(f"{panel['key']:<16} {r['n_red_columns']:>5} {r['line_thickness_px']:>6} "
              f"{r['x'].min():>7.4f}-{r['x'].max():<8.4f} {iv}")
    if args.report_only:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method = (
        "native 300-dpi embedded panel raster -> red-pixel mask "
        f"(R>{RED['r_min']}, G<{RED['g_max']}, B<{RED['b_max']}, "
        f"R-G>{RED['rg_min']}, R-B>{RED['rb_min']}) -> per-column red bands, "
        f"bands separated by <={MERGE_PX} px merged across an overplotted model line "
        "-> loading-branch ordinate (band median outside a reversal interval; median "
        "of the uppermost band's top line-thickness inside one) -> uniform eps_a "
        "resample; axis calibration from the plotted panel frame, checked against "
        "the printed tick marks"
    )

    q_panels, pw_panels = [], []
    for panel in PANELS:
        r = read[panel["key"]]
        n = args.stations if panel["quantity"] == "q" else args.stations
        gx, gy = resample(r["x"], r["y"], n)
        rec = dict(key=panel["key"], test_type=panel["test_type"],
                   specimen_id=panel["specimen_id"], theta_deg=panel["theta_deg"],
                   source_locator=panel["source_locator"],
                   n_raw_red_columns=r["n_red_columns"],
                   n_panel_columns=r["n_panel_columns"],
                   n_resampled_points=int(n),
                   line_thickness_px=r["line_thickness_px"],
                   reversal_intervals_eps_a_pct=r["reversal_intervals"],
                   eps_a_pct_range=[float(gx.min()), float(gx.max())],
                   eps_a_pct=[float(v) for v in gx])
        if panel["quantity"] == "q":
            rec["sigma_r0_MPa"] = panel["sigma_r0_MPa"]
            rec["q_MPa_range"] = [float(gy.min()), float(gy.max())]
            rec["q_MPa"] = [float(v) for v in gy]
            q_panels.append(rec)
        else:
            rec["sigma_r0_eff_MPa"] = panel["sigma_r0_eff_MPa"]
            rec["pw_MPa_range"] = [float(gy.min()), float(gy.max())]
            rec["pw_MPa"] = [float(v) for v in gy]
            rec["pw_max_MPa"] = float(gy.max())
            rec["eps_a_pct_at_pw_max"] = float(gx[int(np.argmax(gy))])
            pw_panels.append(rec)

    common = dict(schema_version="2.0", source_paper=SOURCE_PAPER,
                  source_pdf=str(pdf), digitization_method=method)

    curves = dict(common, source_pages=[6, 7], panels_digitized=q_panels,
                  convention_note=(
                      "theta = angle [deg] between bedding plane and specimen loading "
                      "axis; this is beta in Nordas 2024, where beta = 90 deg minus "
                      "the source bedding dip theta (Sec. 3). S-test theta=90, "
                      "P-test theta=0, Z-test theta=30 (source dip 60 deg in Fig.2c)."),
                  cu_handling_note=(
                      "CU tests (Fig.2 a/b/c) are undrained; effective sigma'_r evolves "
                      "with pore pressure. Reported sigma_r0_MPa = initial effective "
                      "confining pressure ~13 MPa per Nordas 2024 page 8 line 575-577. "
                      "The production lanes drive the surrogates along the measured "
                      "effective stress path built from the pore-pressure panels."),
                  reading_note=(
                      "On CU-P, CD-S and CD-P the plotted record doubles back in axial "
                      "strain near the end of the test, so those strain columns carry "
                      "both the loading branch and the branch followed after the "
                      "reversal. The deviator can only fall across a reversal, so the "
                      "loading branch is the upper one and is what these stations "
                      "carry. CU-S and CU-Z carry no reversal."))
    pore = dict(common, source_pages=[6], panels_digitized=pw_panels,
                quantity="excess pore pressure p_w [MPa] versus axial strain [%]",
                sign_convention=(
                    "p_w positive = pore pressure build-up, so the effective radial "
                    "stress during the undrained test is sigma_r_eff(eps_a) = "
                    "sigma_r0_eff - p_w(eps_a) at constant total radial stress"),
                reading_note=(
                    "Each pore-pressure panel is read on the reversal intervals "
                    "detected on its own specimen's deviator panel: the abscissa is "
                    "the same axial strain, so the record doubles back at the same "
                    "stations."))

    for name, payload in (("digitized_nordas_curves.json", curves),
                          ("digitized_nordas_pore_pressure.json", pore)):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
