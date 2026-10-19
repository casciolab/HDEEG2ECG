# -*- coding: utf-8 -*-
"""
fig_style.py -- house style for the HDEEG2ECG figures.

Import this first in every figure script. It sets a single consistent visual
language so the figures read as one family, and it enforces the things journals
actually reject on: font embedding, vector output, physical width in mm, and
hairline strokes that survive print.

    import fig_style as fs
    fs.use()
    fig = fs.figure(width='double', height_mm=180)
    ...
    fs.save(fig, 'my_figure')

Design decisions worth knowing about
------------------------------------
* Widths are in millimetres, not inches, because that is how journals specify
  them. 89 mm single column, 183 mm double, 120 mm for 1.5-column layouts
  (Nature-family conventions; adjust WIDTHS_MM for your target).
* Fonts are embedded as TrueType (Type 42), not Type 3. Type 3 is what makes a
  PDF un-editable in Illustrator and triggers preflight warnings.
* Font sizes are set so that text is 7 pt AFTER the figure is placed at its
  final size. Never scale a figure in the manuscript -- build it at final size.
* One semantic colour map, used identically in every panel. Reusing a hue for
  two different meanings across panels is the single most common way a
  multi-panel figure becomes unreadable.
* Colours are checked for deuteranopia/protanopia separability. The true vs
  reconstructed pair is additionally distinguished by line style, so the figure
  survives greyscale printing.

@author: figures for Fraile-Vazquez et al.
"""

from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ============================================================
# PHYSICAL SIZE
# ============================================================
MM = 1.0 / 25.4                      # mm -> inches

WIDTHS_MM = {
    'single': 89.0,                  # one column
    'onehalf': 120.0,
    'double': 183.0,                 # full page width
}

# ============================================================
# SEMANTIC PALETTE  (use these names, never raw hex, in figure scripts)
# ============================================================
C = {
    # trigger source -- the paper's central contrast
    'true':        '#1A1A1A',        # ground-truth ECG            (solid line)
    'recon':       '#C8442A',        # reconstructed ECG           (dashed line)
    'surrogate':   '#9A9890',        # time-shifted surrogate      (dotted line)

    # quality-control verdict
    'pass':        '#12796B',        # gate retains
    'fail':        '#B4341F',        # gate rejects
    'threshold':   '#6E6C66',        # frozen decision boundary

    # cohort (used only where a cohort split is the point)
    'aut':         '#3E6B9B',
    'naut':        '#C08A2E',

    # neutral furniture
    'ink':         '#1A1A1A',
    'ink_soft':    '#55534E',
    'ink_faint':   '#8C8A83',
    'rule':        '#C9C7BF',
    'fill_soft':   '#E8E6DF',
}

LINESTYLE = {'true': '-', 'recon': (0, (4.5, 2.0)), 'surrogate': (0, (1.2, 1.8))}

# Diverging map for the carpet plot, centred on zero, neutral (not white) midpoint
# so that zero is visually distinct from missing data.
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'hep_div',
    ['#1C4E77', '#5B8DB8', '#AFC6D8', '#E6E4DD', '#E8B69A', '#C8654A', '#8A2B18'],
    N=256,
)


def diverging_norm(data, symmetric=True):
    """Norm centred on zero. Symmetric by default so colour encodes sign honestly."""
    if symmetric:
        v = float(np.nanpercentile(np.abs(data), 99))
        return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    return TwoSlopeNorm(vmin=float(np.nanmin(data)), vcenter=0.0,
                        vmax=float(np.nanmax(data)))


# ============================================================
# RCPARAMS
# ============================================================
def use(base_pt: float = 7.0, family: str = 'sans-serif'):
    """Apply the house style. Call once, before creating any figure.

    base_pt is the size of tick labels at final print size. Axis labels are
    base+0.5, panel letters are base+2. Seven point is the floor most journals
    accept; do not go below it to fit more in.
    """
    mpl.rcParams.update({
        # --- output ---
        'pdf.fonttype': 42,           # embed TrueType, keep text editable
        'ps.fonttype': 42,
        'svg.fonttype': 'none',       # keep text as text in SVG
        'figure.dpi': 150,            # screen preview
        'savefig.dpi': 600,           # raster fallback, print quality
        'savefig.bbox': 'standard',   # NOT 'tight' -- tight breaks exact mm width
        'savefig.transparent': False,
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        # --- typography ---
        'font.family': family,
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': base_pt,
        'axes.labelsize': base_pt + 0.5,
        'axes.titlesize': base_pt + 0.5,
        'xtick.labelsize': base_pt - 0.5,
        'ytick.labelsize': base_pt - 0.5,
        'legend.fontsize': base_pt - 0.5,
        'mathtext.default': 'regular',
        'axes.titlelocation': 'left',
        'axes.titlepad': 3.0,
        'axes.labelpad': 2.0,

        # --- strokes: hairlines, print-safe ---
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        'xtick.minor.size': 1.3,
        'ytick.minor.size': 1.3,
        'xtick.major.pad': 1.8,
        'ytick.major.pad': 1.8,
        'lines.linewidth': 0.9,
        'lines.markersize': 3.0,
        'patch.linewidth': 0.5,
        'grid.linewidth': 0.4,
        'hatch.linewidth': 0.4,

        # --- chrome: minimal ---
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.edgecolor': C['ink'],
        'axes.labelcolor': C['ink'],
        'text.color': C['ink'],
        'xtick.color': C['ink_soft'],
        'ytick.color': C['ink_soft'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'legend.labelspacing': 0.3,
        'legend.borderpad': 0.0,
        'legend.columnspacing': 1.0,

        # --- images ---
        'image.interpolation': 'nearest',
        'image.origin': 'lower',
        'image.cmap': 'RdBu_r',
    })


def figure(width='double', height_mm=120.0, **kw):
    """Create a figure at an exact physical size.

    width : 'single' | 'onehalf' | 'double' | float (mm)
    """
    w_mm = WIDTHS_MM.get(width, width) if isinstance(width, str) else float(width)
    return plt.figure(figsize=(w_mm * MM, height_mm * MM), **kw)


# ============================================================
# ANNOTATION HELPERS
# ============================================================
def panel_label(ax, letter, dx=-0.055, dy=1.06, weight='bold', size=None):
    """Panel letter in the axes' own coordinate frame.

    Placed relative to the axes rather than the figure so it stays put when the
    layout is adjusted. Lowercase and bold is the most common house style; some
    journals want uppercase, so change it in one place here.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=size or (mpl.rcParams['font.size'] + 2),
            fontweight=weight, va='baseline', ha='right', color=C['ink'])


def despine(ax, keep=('left', 'bottom')):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)


def offset_spines(ax, pts=3.0):
    """Detach the axes from the data. Reads as more considered than a closed box."""
    for side in ('left', 'bottom'):
        if ax.spines[side].get_visible():
            ax.spines[side].set_position(('outward', pts))


def scalebar(ax, x, y, dx, dy, xlabel, ylabel, lw=0.9, pad=0.012):
    """L-shaped scale bar. Use instead of axes when the absolute values on an
    axis are not the point (raw traces, butterfly plots) -- it removes an entire
    axis worth of chrome while keeping the units legible."""
    ax.plot([x, x, x + dx], [y + dy, y, y], color=C['ink'], lw=lw,
            solid_capstyle='butt', clip_on=False, zorder=6)
    ax.text(x + dx / 2, y - pad * (ax.get_ylim()[1] - ax.get_ylim()[0]), xlabel,
            ha='center', va='top', color=C['ink_soft'], clip_on=False)
    ax.text(x - pad * (ax.get_xlim()[1] - ax.get_xlim()[0]), y + dy / 2, ylabel,
            ha='right', va='center', rotation=90, color=C['ink_soft'], clip_on=False)


def stat_text(ax, s, x=0.98, y=0.96, ha='right', va='top', color=None):
    """Inline statistic. Put numbers inside the panel, not in the caption --
    a reader should not have to look away to know what they are seeing."""
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va,
            color=color or C['ink_soft'], linespacing=1.35)


def shade_window(ax, t0, t1, color=None, alpha=0.10, zorder=0):
    ax.axvspan(t0, t1, color=color or C['fill_soft'], alpha=alpha,
               lw=0, zorder=zorder)


def zeroline(ax, axis='y', **kw):
    kw.setdefault('color', C['rule'])
    kw.setdefault('lw', 0.5)
    (ax.axhline if axis == 'y' else ax.axvline)(0, **kw)


# ============================================================
# SAVE
# ============================================================
def save(fig, stem, outdir=None, formats=('pdf', 'png', 'svg'),
         verbose=True):
    """Write vector + raster copies. PDF is the submission format; SVG is for
    hand-editing in Illustrator/Inkscape; PNG is for slides and preprints."""
    outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in formats:
        p = os.path.join(outdir, f'{stem}.{ext}')
        fig.savefig(p, format=ext)
        paths.append(p)
    if verbose:
        w_in, h_in = fig.get_size_inches()
        print(f'  saved {stem}: {w_in/MM:.1f} x {h_in/MM:.1f} mm -> '
              + ', '.join(os.path.basename(p) for p in paths))
    return paths


def check_text_sizes(fig, floor_pt=6.0):
    """Warn about any text below the journal floor. Run before submitting --
    it catches the tick label that ended up at 5 pt because of an autoscale."""
    small = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_text().strip() and t.get_fontsize() < floor_pt:
            small.append((t.get_text()[:30], round(t.get_fontsize(), 2)))
    if small:
        print(f'  WARNING: {len(small)} text elements below {floor_pt} pt:')
        for s, sz in small[:10]:
            print(f'    {sz} pt  {s!r}')
    else:
        print(f'  all text >= {floor_pt} pt')
    return small