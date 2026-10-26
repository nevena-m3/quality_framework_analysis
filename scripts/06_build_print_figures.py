"""Redraw manuscript Figs. 2-4 at their printed size (JBHI full page width).

The notebooks draw each figure on a 7.35-inch canvas that is then scaled down to
4-5 inches in the manuscript, so the printed text ends up at about 4 pt. This
script redraws the same panels directly at 7.16 inches with text of 8 pt or
larger. It reads only the figure source tables that notebooks 02, 03 and 06
already export (summary statistics, no participant-level rows), so no analysis
is rerun and no plotted value changes.

Usage:
    python scripts/06_build_print_figures.py
    python scripts/06_build_print_figures.py --source-root path/to/unzipped/tables --out figures_print

`--source-root` defaults to the repository root (or PAPER1_ANALYSIS_ROOT). Input
files are located by name below it, so an unzipped copy of the source_data
folders works as well as the original outputs/ tree.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator  # noqa: E402

# ------------------------------------------------------------------
# Print styling: JBHI text width 7.16 in, all text >= 8 pt.
# Ink for text, hairline grid and axes, colour only where it carries meaning.
# ------------------------------------------------------------------
PAGE_WIDTH = 7.16
FS = 8.0          # ticks, labels, legends
FS_TITLE = 8.5    # panel titles
FS_LETTER = 10.0  # panel letters

INK = '#111111'     # text, spines, ticks
INK2 = '#333333'    # axis labels
MUTED = '#9fb0d8'   # open markers in scatter
AXIS = '#111111'    # spines and ticks (Nature style: thin black axes)
GRID = '#ececec'    # unused by default; kept for optional grids
BAND = '#ffffff'    # no row bands
RULE = '#111111'    # header rules
CONNECT = '#c9d1e3' # dumbbell connectors
ACCENT = '#E64B35'  # NPG red: ALS, within-participant, positive agreement

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
    'font.size': FS,
    'text.color': INK,
    'axes.titlesize': FS_TITLE,
    'axes.labelsize': FS,
    'axes.labelcolor': INK2,
    'axes.edgecolor': AXIS,
    'axes.facecolor': 'none',
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'xtick.color': AXIS,
    'ytick.color': AXIS,
    'xtick.labelcolor': INK2,
    'ytick.labelcolor': INK2,
    'legend.fontsize': FS,
    'legend.frameon': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.pad': 2.0,
    'ytick.major.pad': 2.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# Nature Publishing Group style. Family identity: navy, green, gold, raspberry (validated as a
# categorical set;
# every use is also labelled, so colour never carries identity alone).
CONTROL_COLOR, ALS_COLOR = '#4DBBD5', ACCENT
BETWEEN_COLOR, WITHIN_COLOR = '#3C5488', ACCENT
FAMILY_ORDER = ['QADD', 'QGAIN', 'QREV', 'QCHAN']
FAMILY_COLORS = {'QADD': '#3C5488', 'QGAIN': '#00A087', 'QREV': '#E5A019', 'QCHAN': '#B5446E',
                 'QDIST': '#7E6148', 'QTEMP': '#7E6148'}
FEAT_FAMILY: dict = {}  # feature -> family, filled by the builders
ROW_GAP = 0.65  # vertical gap between families, in row units

# Diverging navy - white - NPG red, equal steps per arm.
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    'blue_grey_red', ['#1d2f5e', '#3C5488', '#a3b2d9', '#f7f7f7', '#f4a796', '#E64B35', '#8a1c0f'])
DIVERGING.set_bad((0, 0, 0, 0))


# Alternative palettes (choose with --palette); each sets the cohort pair, the four family colours
# and the diverging heatmap scale. The default above is the NPG palette.
PALETTES = {
    'midnight': dict(control='#1B2A49', als='#B03A55', fam=['#1B2A49', '#1F7A8C', '#C08A1E', '#8E2F63'],
                     div=['#101a30', '#1B2A49', '#6E86B8', '#FAF9F6', '#E08A9B', '#B03A55', '#5E1427']),
    'petrol': dict(control='#0F4C5C', als='#C2551F', fam=['#155E8A', '#3F7F2A', '#D08E12', '#8E1F3C'],
                   div=['#07303a', '#0F4C5C', '#5A9AAA', '#FAF9F6', '#E89A72', '#C2551F', '#6E2a0c']),
    'oxford': dict(control='#1E3560', als='#8C1C3A', fam=['#1E3560', '#1F7568', '#B5800A', '#8C1C3A'],
                   div=['#101f3d', '#1E3560', '#6F86B5', '#FAF9F6', '#CC7088', '#8C1C3A', '#4a0a1c']),
    'ocean': dict(control='#005F73', als='#AE2012', fam=['#005F73', '#0A9396', '#EE9B00', '#AE2012'],
                  div=['#003945', '#005F73', '#4FA3B0', '#FAF9F6', '#E07A6C', '#AE2012', '#5F0F08']),
    'slate': dict(control='#28365E', als='#7A2F78', fam=['#28365E', '#2F7D62', '#CF8A1F', '#7A2F78'],
                  div=['#161e38', '#28365E', '#7484B4', '#FAF9F6', '#B97DB6', '#7A2F78', '#3f1340']),
}


def apply_palette(name: str) -> None:
    pal = PALETTES[name]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, pal['div'])
    cmap.set_bad((0, 0, 0, 0))
    other = tint(pal['control'], 0.35)
    globals().update(
        ACCENT=pal['als'], CONTROL_COLOR=pal['control'], ALS_COLOR=pal['als'],
        BETWEEN_COLOR=pal['control'], WITHIN_COLOR=pal['als'], DIVERGING=cmap,
        MUTED=tint(pal['control'], 0.6), CONNECT=tint(pal['control'], 0.8),
        FAMILY_COLORS={**dict(zip(FAMILY_ORDER, pal['fam'])), 'QDIST': other, 'QTEMP': other})


def find(root: Path, name: str, required: bool = True) -> Path | None:
    direct = [p for p in root.rglob(name) if p.is_file() and 'checkpoint' not in str(p).lower()]
    if not direct:
        if required:
            raise FileNotFoundError(f'{name} not found below {root}')
        return None
    return sorted(direct, key=lambda p: len(str(p)))[0]


def read(root: Path, name: str, required: bool = True) -> pd.DataFrame | None:
    path = find(root, name, required)
    return pd.read_csv(path, low_memory=False) if path is not None else None


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def tidy(label) -> str:
    """House style for measure names: en dash in 'speech–pause', no trailing 'present'."""
    text = str(label).replace(' present', '').strip()
    return text.replace('Speech-pause', 'Speech–pause').replace('speech-pause', 'speech–pause')


def head(ax, letter, title='', dx=-14, dy=5, gap=11):
    """Panel letter and title on one baseline, just above the axes' top-left corner."""
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(dx, dy), textcoords='offset points',
                fontsize=FS_LETTER, fontweight='bold', ha='left', va='baseline', color=INK)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords='axes fraction', xytext=(dx + gap, dy),
                    textcoords='offset points', fontsize=FS_TITLE, fontweight='bold', ha='left',
                    va='baseline', color=INK)


def or_axis(ax, values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    lo, hi = (0.5, 2.0) if clean.size == 0 else (min(0.5, clean.min() / 1.25), max(2.0, clean.max() * 1.25))
    ax.set_xscale('log')
    ax.set_xlim(lo / 1.2, hi * 1.2)
    ticks = [t for t in [0.25, 0.5, 1, 2, 4, 8] if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def forest(ax, frame, positions, est, low, high, star=True, tier_col='inference_tier'):
    """Estimate and 95% CI in the family colour. Filled marker + star = q < 0.05; open = not significant
    (all filled where no test applies, e.g. ICC)."""
    for _, row in frame.iterrows():
        if row['feature'] not in positions:
            continue
        e, lo, hi = row.get(est, np.nan), row.get(low, np.nan), row.get(high, np.nan)
        if not np.isfinite(e):
            continue
        y = positions[row['feature']]
        q = row.get('q_value', np.nan)
        sig = star and np.isfinite(q) and q < 0.05
        base = FAMILY_COLORS.get(FEAT_FAMILY.get(row['feature']), INK)
        color = base
        face = color if (sig or not star) else 'white'
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle='butt', zorder=3)
            if sig:
                ax.text(hi, y, '*', ha='left', va='center', fontsize=FS + 1, color=INK,
                        transform=ax.transData + mpl.transforms.ScaledTranslation(1.5 / 72, -1.5 / 72,
                                                                                   ax.figure.dpi_scale_trans))
        ax.plot([e], [y], 'o', ms=4.6, mfc=face, mec=color, mew=1.1, zorder=4)


def style_rows(ax, ylim, xgrid=False, null=None):
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines['bottom'].set_position(('outward', 3))
    ax.spines['left'].set_visible(False)
    if xgrid:
        ax.grid(axis='x', color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    if null is not None:
        ax.axvline(null, color=INK, lw=0.5, ls=(0, (4, 2)), zorder=1.5)


def bands(fig, ax, rows, x0, x1, half=0.5):
    """Light full-width bands behind the given row positions (disabled in the Nature style)."""
    if BAND.lower() == '#ffffff':
        return
    to_fig = ax.transData + fig.transFigure.inverted()
    for y in rows:
        _, ya = to_fig.transform((0, y - half))
        _, yb = to_fig.transform((0, y + half))
        fig.patches.append(mpl.patches.Rectangle((x0, min(ya, yb)), x1 - x0, abs(yb - ya),
                                                 transform=fig.transFigure, color=BAND, lw=0, zorder=-10))


def family_rail(ax, fam, y0, y1, x=0.0, name_x=0.06, name=True):
    """Family name in its colour, centred on the family's rows, with a thin bracket."""
    ax.plot([x, x], [y0 - 0.32, y1 + 0.32], color=FAMILY_COLORS.get(fam, INK), lw=0.8,
            solid_capstyle='butt', clip_on=False)
    if name:
        ax.text(name_x, (y0 + y1) / 2, fam, ha='left', va='center', fontweight='bold',
                color=FAMILY_COLORS.get(fam, INK))


def tint(color, amount=0.5):
    """Mix a colour with white (amount = share of white)."""
    r, g, b = mpl.colors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def grouped_positions(families):
    """Row positions with a ROW_GAP between consecutive families."""
    out, y, prev = [], 0.0, None
    for f in families:
        if prev is not None and f != prev:
            y += ROW_GAP
        out.append(y)
        y += 1
        prev = f
    return out


def gapped_edges(sizes, gap=0.32):
    """Cell edges and centres for a heatmap whose family blocks are separated by a small gap."""
    edges, centres, real = [0.0], [], []
    for gi, size in enumerate(sizes):
        if gi:
            edges.append(edges[-1] + gap)
            real.append(None)
        for _ in range(size):
            centres.append(edges[-1] + 0.5)
            real.append(len(centres) - 1)
            edges.append(edges[-1] + 1.0)
    return np.array(edges), np.array(centres), real


def expand(m, real_rows, real_cols):
    out = np.full((len(real_rows), len(real_cols)), np.nan)
    for i, r in enumerate(real_rows):
        for j, c in enumerate(real_cols):
            if r is not None and c is not None:
                out[i, j] = m[r, c]
    return out


def thin_colorbar(fig, img, cax, label, ticks, extend='neither'):
    cb = fig.colorbar(img, cax=cax, extend=extend, extendfrac=0.06)
    cb.outline.set_visible(False)
    cb.set_ticks(ticks)
    cb.ax.tick_params(length=2, width=0.5, color=AXIS)
    cb.set_label(label, labelpad=3, color=INK2)
    return cb


# ==================================================================
# Fig. 2: cohort distributions, persistence, longitudinal change,
#         cohort and bulbar associations (one shared row axis).
# ==================================================================
def build_fig2(root: Path, out: Path) -> Path:
    reg = read(root, 'feature_registry.csv')
    for col in ['include_main']:
        if col in reg:
            reg[col] = as_bool(reg[col])
    std = read(root, 'panel_A_continuous_standardized.csv')
    states = read(root, 'panel_A_states.csv')
    sparse = read(root, 'panel_A_sparse.csv')
    pers = read(root, 'panel_B_persistence.csv')
    long_c = read(root, 'panel_B_longitudinal_continuous.csv')
    long_p = read(root, 'panel_B_longitudinal_presence.csv')
    coh_c = read(root, 'panel_C_cohort_continuous.csv')
    coh_p = read(root, 'panel_C_cohort_presence.csv')
    bul_c = read(root, 'panel_C_bulbar_continuous.csv')
    bul_p = read(root, 'panel_C_bulbar_presence.csv')

    main = reg.loc[reg['include_main']] if 'include_main' in reg else reg
    cont = main.loc[main['outcome_class'].isin(['continuous', 'censored_continuous'])
                    & main['inference_tier'].isin(['core', 'secondary'])]
    cont = cont.loc[cont['feature'].isin(std['feature'])]
    one = main.loc[main['outcome_class'].eq('one_sided_zero')
                   & main['inference_tier'].isin(['core', 'secondary'])]
    spr = main.loc[main['outcome_class'].isin(['sparse_event_extent', 'monitoring'])]
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER + ['QDIST', 'QTEMP'])}
    sort = lambda f: f.assign(_r=f['family'].map(fam_rank)).sort_values(['_r', 'feature_order'])  # noqa: E731
    cont, one, spr = sort(cont), sort(one), sort(spr)
    tier = reg.set_index('feature')['inference_tier'].to_dict()
    FEAT_FAMILY.update(reg.set_index('feature')['family'].to_dict())
    for frame in [pers, long_c, long_p, coh_c, coh_p, bul_c, bul_p]:
        if 'inference_tier' not in frame:
            frame['inference_tier'] = frame['feature'].map(tier)

    # Continuous rows, grouped by family.
    ys = grouped_positions(cont['family'].tolist())
    pos = dict(zip(cont['feature'], ys))
    fam_span = {}
    for f, yv in zip(cont['family'], ys):
        fam_span.setdefault(f, []).append(yv)
    ylim = (ys[-1] + 0.6, -0.6)
    n_rows = ys[-1] + 1

    # Binary strip: two QCHAN deficits, then sparse QDIST/QTEMP events.
    strip_rows = [(f, lbl, fam, t) for f, lbl, fam, t in
                  zip(one['feature'], one['short_label'], one['family'], one['inference_tier'])]
    strip_rows += [(f, lbl, fam, 'secondary') for f, lbl, fam in zip(spr['feature'], spr['short_label'], spr['family'])]
    sys_ = grouped_positions([r[2] for r in strip_rows])
    spos = {r[0]: yv for r, yv in zip(strip_rows, sys_)}
    sylim = (sys_[-1] + 0.6, -0.6)

    row_h = 0.152
    top_h, mid_h = 0.66, 0.66
    cont_h, strip_h = row_h * (n_rows + 0.2), row_h * (sys_[-1] + 1.2)
    fig_h = top_h + cont_h + mid_h + strip_h + 0.52
    fig = plt.figure(figsize=(PAGE_WIDTH, fig_h))
    widths = [1.72, 0.90, 0.64, 0.75, 0.72, 0.70, 0.70]
    left, right = 0.04, PAGE_WIDTH - 0.06
    gaps = (right - left - sum(widths)) / (len(widths) - 1)
    xs, x = [], left
    for w in widths:
        xs.append((x / PAGE_WIDTH, w / PAGE_WIDTH))
        x += w + gaps
    y_cont = (fig_h - top_h - cont_h) / fig_h
    y_strip = 0.52 / fig_h
    h_cont, h_strip = cont_h / fig_h, strip_h / fig_h

    def axes_pair(i):
        a = fig.add_axes([xs[i][0], y_cont, xs[i][1], h_cont])
        b = fig.add_axes([xs[i][0], y_strip, xs[i][1], h_strip])
        return a, b

    # Labels column with family rails.
    lab, slab = axes_pair(0)
    for ax, lim in [(lab, ylim), (slab, sylim)]:
        ax.set_xlim(0, 1)
        ax.set_ylim(*lim)
        ax.axis('off')
    for fam, yy in fam_span.items():
        family_rail(lab, fam, yy[0], yy[-1], name_x=0.04)
    for r in cont.itertuples(index=False):
        lab.text(0.32, pos[r.feature], tidy(r.short_label), ha='left', va='center',
                 fontweight='bold' if r.inference_tier == 'core' else 'normal')
    fam_strip = {}
    for f, lbl, fam, t in strip_rows:
        fam_strip.setdefault(fam, []).append(spos[f])
        slab.text(0.32, spos[f], tidy(lbl), ha='left', va='center',
                  fontweight='bold' if t == 'core' else 'normal')
    for fam, yy in fam_strip.items():
        family_rail(slab, fam, yy[0], yy[-1], name_x=0.04)
    x_band0, x_band1 = xs[0][0] + 0.03 / PAGE_WIDTH, xs[-1][0] + xs[-1][1]
    for ax, spans in [(lab, fam_span), (slab, fam_strip)]:
        bands(fig, ax, [yy[k] for yy in spans.values() for k in range(0, len(yy), 2)], x_band0, x_band1)

    # A: distributions (continuous) and prevalence (strip).
    ax, sx = axes_pair(1)
    for r in cont.itertuples(index=False):
        for cohort, off, col in [('CONTROL', -0.17, CONTROL_COLOR), ('ALS', 0.17, ALS_COLOR)]:
            s = std.loc[(std['feature'] == r.feature) & (std['cohort'] == cohort)]
            if s.empty:
                continue
            s = s.iloc[0]
            yy = pos[r.feature] + off
            ax.plot([s['p05_z'], s['p95_z']], [yy, yy], color=col, lw=0.6, alpha=0.9, solid_capstyle='butt')
            ax.plot([s['q1_z'], s['q3_z']], [yy, yy], color=col, lw=3.2, solid_capstyle='butt')
            ax.plot([s['median_z']], [yy], 'o', ms=3.4, mfc='white', mec=col, mew=1.0, zorder=3)
    style_rows(ax, ylim, null=0)
    ax.set_xlim(-3.2, 3.2)
    ax.set_xticks([-3, 0, 3])
    ax.set_xlabel('Transformed value\n(IQR units)', labelpad=2)
    lab.legend(handles=[
        mpl.lines.Line2D([0], [0], color=CONTROL_COLOR, lw=3.2, label='Control'),
        mpl.lines.Line2D([0], [0], color=ALS_COLOR, lw=3.2, label='ALS')],
        ncol=2, loc='lower left', bbox_to_anchor=(0.0, 1.0), handlelength=1.3,
        columnspacing=1.0, borderaxespad=0.6, handletextpad=0.4)
    for f, *_ in strip_rows:
        for cohort, off, col in [('CONTROL', -0.17, CONTROL_COLOR), ('ALS', 0.17, ALS_COLOR)]:
            if f in set(one['feature']):
                rec = states.loc[(states['feature'] == f) & (states['cohort'] == cohort)]
                val = 100 - float(rec.iloc[0]['zero_percent_available']) if not rec.empty else np.nan
            else:
                rec = sparse.loc[(sparse['feature'] == f) & (sparse['cohort'] == cohort)]
                val = (100 * float(rec.iloc[0]['positive_count']) / float(rec.iloc[0]['n_available_recordings'])
                       if not rec.empty and float(rec.iloc[0]['n_available_recordings']) > 0 else np.nan)
            if np.isfinite(val):
                sx.plot([0, val], [spos[f] + off] * 2, color=col, lw=3.2, solid_capstyle='butt', zorder=3)
                sx.plot([val], [spos[f] + off], 'o', ms=2.6, color=col, mew=0, zorder=4)
    style_rows(sx, sylim)
    sx.set_xlim(0, 100)
    sx.set_xticks([0, 50, 100])
    sx.set_xlabel('Recordings (%)', labelpad=2)

    # B: persistence and annual change.
    ax, sx = axes_pair(2)
    icc = pers.rename(columns={'icc': 'e', 'icc_ci95_low': 'l', 'icc_ci95_high': 'h'})
    forest(ax, icc, pos, 'e', 'l', 'h', star=False)
    forest(sx, icc, spos, 'e', 'l', 'h', star=False)
    for a, lim in [(ax, ylim), (sx, sylim)]:
        style_rows(a, lim)
        a.set_xlim(-0.05, 1.05)
        a.set_xticks([0, 0.5, 1])
        a.set_xticklabels(['0', '0.5', '1'])
        a.set_xlabel('ICC', labelpad=2)
    ax.set_title('Persistence', loc='left', pad=4, color=INK2)

    ax, sx = axes_pair(3)
    forest(ax, long_c, pos, 'estimate', 'ci95_low', 'ci95_high')
    style_rows(ax, ylim, null=0)
    ax.set_title('Annual change', loc='left', pad=4, color=INK2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
    ax.set_xlabel('IQR units\nper year', labelpad=2)
    lp = long_p.copy()
    if not lp.empty:
        lp['o'], lp['ol'], lp['oh'] = np.exp(lp['estimate']), np.exp(lp['ci95_low']), np.exp(lp['ci95_high'])
        forest(sx, lp, spos, 'o', 'ol', 'oh')
        or_axis(sx, pd.concat([lp['ol'], lp['oh']]))
    style_rows(sx, sylim, null=1)
    sx.set_xlabel('Odds ratio\nper year', labelpad=2)

    # C: cohort difference and bulbar associations (per point of worse bulbar function).
    ax, sx = axes_pair(4)
    forest(ax, coh_c, pos, 'estimate', 'ci95_low', 'ci95_high')
    style_rows(ax, ylim, null=0)
    ax.set_title('ALS − control', loc='left', pad=4, color=INK2)
    ax.set_xlabel('IQR units', labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
    cp = coh_p.copy()
    cp['o'], cp['ol'], cp['oh'] = np.exp(cp['estimate']), np.exp(cp['ci95_low']), np.exp(cp['ci95_high'])
    forest(sx, cp, spos, 'o', 'ol', 'oh')
    or_axis(sx, pd.concat([cp['ol'], cp['oh']]))
    style_rows(sx, sylim, null=1)
    sx.set_xlabel('Odds ratio', labelpad=2)

    for i, term, title in [(5, 'bulbar_between', 'Between'), (6, 'bulbar_within', 'Within')]:
        ax, sx = axes_pair(i)
        f = bul_c.loc[bul_c['target_term'].eq(term)].copy()
        f['d'], f['dl'], f['dh'] = -f['estimate'], -f['ci95_high'], -f['ci95_low']
        forest(ax, f, pos, 'd', 'dl', 'dh')
        style_rows(ax, ylim, null=0)
        ax.set_title(title, loc='left', pad=4, color=INK2)
        ax.set_xlabel('IQR units per\npoint worse', labelpad=2)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
        p = bul_p.loc[bul_p['target_term'].eq(term)].copy()
        p['o'], p['ol'], p['oh'] = np.exp(-p['estimate']), np.exp(-p['ci95_high']), np.exp(-p['ci95_low'])
        forest(sx, p, spos, 'o', 'ol', 'oh')
        or_axis(sx, pd.concat([p['ol'], p['oh']]))
        style_rows(sx, sylim, null=1)
        sx.set_xlabel('Odds ratio per\npoint worse', labelpad=2)

    # Group headers: letter + title with a hairline rule spanning the group's columns.
    y_hdr = (fig_h - 0.21) / fig_h
    for i, j, L, text in [(1, 1, 'A', 'Distribution'), (2, 3, 'B', 'Persistence and change'),
                          (4, 6, 'C', 'Cohort and bulbar associations')]:
        x0, x1 = xs[i][0], xs[j][0] + xs[j][1]
        fig.text(x0, y_hdr, L, fontsize=FS_LETTER, fontweight='bold', ha='left', va='baseline')
        fig.text(x0 + 0.16 / PAGE_WIDTH, y_hdr, text, fontsize=FS_TITLE, fontweight='bold', ha='left',
                 va='baseline')
        fig.add_artist(mpl.lines.Line2D([x0, x1], [y_hdr - 0.07 / fig_h] * 2, color=RULE, lw=0.6))
    # Sub-header over the two bulbar columns.
    x0, x1 = xs[5][0], xs[6][0] + xs[6][1]
    y_sub = (fig_h - 0.40) / fig_h
    fig.text((x0 + x1) / 2, y_sub, 'Bulbar severity', ha='center', va='baseline', color=INK2)
    fig.add_artist(mpl.lines.Line2D([x0, x1], [y_sub - 0.05 / fig_h] * 2, color=RULE, lw=0.5))
    return save(fig, out, 'fig2')


# ==================================================================
# Fig. 3: structure among quality measures.
# ==================================================================
def build_fig3(root: Path, out: Path) -> Path:
    key = read(root, 'main_feature_key.csv')
    pairs = read(root, 'main_pairwise_structure.csv')
    comps = read(root, 'main_pca_components.csv')
    loads = read(root, 'main_pca_loadings.csv')
    key['_r'] = key['family'].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    key = key.sort_values(['_r', 'feature_order'])
    order = key['feature'].tolist()
    codes = key.set_index('feature')['feature_code'].to_dict()
    fam = key.set_index('feature')['family'].to_dict()
    for c in ['same_family', 'structural_dependency']:
        pairs[c] = as_bool(pairs[c])
    comps['retained_stable_component'] = as_bool(comps['retained_stable_component'])

    def matrix(col):
        idx = {f: i for i, f in enumerate(order)}
        m = np.full((len(order), len(order)), np.nan)
        np.fill_diagonal(m, 1.0)
        for r in pairs.itertuples(index=False):
            if r.feature_x in idx and r.feature_y in idx:
                m[idx[r.feature_x], idx[r.feature_y]] = m[idx[r.feature_y], idx[r.feature_x]] = getattr(r, col)
        return m

    fig_h = 6.85
    fig = plt.figure(figsize=(PAGE_WIDTH, fig_h))

    def ax_in(x, y, w, h):  # inches from bottom-left
        return fig.add_axes([x / PAGE_WIDTH, y / fig_h, w / PAGE_WIDTH, h / fig_h])

    sizes = [sum(1 for f in order if fam[f] == F) for F in FAMILY_ORDER]
    edges, centres, real = gapped_edges(sizes)
    n = len(order)
    lower = np.tril(np.ones((n, n), dtype=bool))
    hm = 2.60
    img = None
    for x0, col, L, title in [(0.50, 'between_rho', 'A', 'Between participants'),
                              (3.62, 'within_rho', 'B', 'Within participants')]:
        ax = ax_in(x0, 3.87, hm, hm)
        m = matrix(col)
        shown = np.where(lower, m, np.nan)
        img = ax.pcolormesh(edges, edges, np.ma.masked_invalid(expand(shown, real, real)), cmap=DIVERGING,
                            vmin=-0.8, vmax=0.8, edgecolors='white', linewidth=0.35)
        for i in range(n):
            for j in range(i + 1):
                if not np.isfinite(m[i, j]):
                    ax.add_patch(mpl.patches.Rectangle((centres[j] - 0.5, centres[i] - 0.5), 1, 1,
                                                       facecolor='white', edgecolor=MUTED, hatch='//////',
                                                       lw=0.0))
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[-1], edges[0])
        ax.set_aspect('equal')
        ax.set_xticks(centres)
        ax.set_yticks(centres)
        ax.set_xticklabels([codes[f] for f in order], rotation=90)
        ax.set_yticklabels([codes[f] for f in order])
        ax.tick_params(length=0, pad=6)
        for s in ax.spines.values():
            s.set_visible(False)
        start = 0
        for F, size in zip(FAMILY_ORDER, sizes):
            lo, hi = centres[start] - 0.5, centres[start + size - 1] + 0.5
            c = FAMILY_COLORS[F]
            ax.add_patch(mpl.patches.Rectangle((-0.55, lo), 0.32, hi - lo, color=c, clip_on=False, lw=0))
            ax.add_patch(mpl.patches.Rectangle((lo, edges[-1] + 0.23), hi - lo, 0.32, color=c, clip_on=False,
                                               lw=0))
            start += size
        head(ax, L, title, dx=-24)
    cax = ax_in(6.62, 4.45, 0.09, 1.70)
    thin_colorbar(fig, img, cax, 'Rank correlation, ρ', [-0.8, -0.4, 0, 0.4, 0.8], extend='both')

    # C: between vs within scatter.
    ax = ax_in(0.55, 1.43, 1.75, 1.75)
    d = pairs.loc[pairs['between_rho'].notna() & pairs['within_rho'].notna()]
    bf = d.loc[~d['same_family'] & ~d['structural_dependency']]
    wf = d.loc[d['same_family'] & ~d['structural_dependency']]
    sd = d.loc[d['structural_dependency']]
    ax.axhline(0, color=CONNECT, lw=0.5, zorder=0)
    ax.axvline(0, color=CONNECT, lw=0.5, zorder=0)
    ax.plot([-1, 1], [-1, 1], ls=(0, (3, 2)), color=INK2, lw=0.6, zorder=1)
    ax.scatter(bf['between_rho'], bf['within_rho'], s=11, facecolors='none', edgecolors=MUTED, lw=0.6,
               label='Between-family', zorder=2)
    ax.scatter(wf['between_rho'], wf['within_rho'], s=13, color=BETWEEN_COLOR, lw=0, label='Within-family', zorder=3)
    ax.scatter(sd['between_rho'], sd['within_rho'], s=24, marker='D', color=ACCENT, edgecolor='white',
               lw=0.5, label='Mechanically related', zorder=4)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect('equal')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(['−1', '', '0', '', '1'])
    ax.set_yticklabels(['−1', '', '0', '', '1'])
    ax.set_xlabel('Between-participant ρ', labelpad=2)
    ax.set_ylabel('Within-participant ρ', labelpad=1)
    ax.legend(loc='lower right', handletextpad=0.1, borderaxespad=0.1, labelspacing=0.3)
    head(ax, 'C', 'Between vs. within', dx=-30)

    # D: scree with parallel analysis.
    ax = ax_in(2.92, 1.43, 1.42, 1.75)
    k = comps.groupby('level')['component'].max().max()
    for level, mk, col in [('between', 'o', BETWEEN_COLOR), ('within', 's', WITHIN_COLOR)]:
        c = comps.loc[comps['level'] == level].sort_values('component')
        ax.plot(c['component'], 100 * c['parallel_95th_eigenvalue'] / k, ls=(0, (3, 2)), lw=0.8, color=col,
                alpha=0.7, zorder=2)
        ax.plot(c['component'], c['variance_explained_percent'], marker=mk, ms=3.4, lw=1.0, color=col,
                mec='white', mew=0.4, label=level.capitalize(), zorder=3)
    ax.plot([], [], ls=(0, (3, 2)), lw=0.8, color=INK2, label='Null, 95th percentile')
    lo_v, hi_v = ax.get_ylim()
    ax.set_ylim(lo_v, hi_v + 0.45 * (hi_v - lo_v))
    ax.set_xticks(range(1, int(k) + 1))
    ax.set_xlabel('Component', labelpad=2)
    ax.set_ylabel('Variance explained (%)', labelpad=2)
    ax.legend(loc='upper right', handlelength=1.6, borderaxespad=0.1, labelspacing=0.3)
    head(ax, 'D', 'Parallel analysis', dx=-28)

    # E, F: retained loadings (family gaps as in A/B).
    pca_feats = (loads.drop_duplicates('feature').assign(_r=lambda f: f['feature'].map(
        {ft: i for i, ft in enumerate(order)})).sort_values('_r'))
    pfam = [fam.get(f, 'QADD') for f in pca_feats['feature']]
    psizes = [pfam.count(F) for F in FAMILY_ORDER if F in pfam]
    pf_present = [F for F in FAMILY_ORDER if F in pfam]
    pedges, pcentres, preal = gapped_edges(psizes)
    pcodes = pca_feats['feature_code'].tolist()
    limg = None
    for x0, level, L, title in [(4.80, 'between', 'E', 'Between'), (5.64, 'within', 'F', 'Within')]:
        keep = comps.loc[(comps['level'] == level) & comps['retained_stable_component'], 'component'].tolist()
        ax = ax_in(x0, 1.43, 0.19 * max(len(keep), 1), 1.75)
        if keep:
            piv = (loads.loc[(loads['level'] == level) & loads['component'].isin(keep)]
                   .pivot(index='feature_code', columns='component', values='loading').reindex(pcodes))
            vals = expand(piv.to_numpy(dtype=float), preal, list(range(len(keep))))
            limg = ax.pcolormesh(np.arange(len(keep) + 1), pedges, np.ma.masked_invalid(vals), cmap=DIVERGING,
                                 vmin=-0.8, vmax=0.8, edgecolors='white', linewidth=0.35)
            ax.set_xticks(np.arange(len(keep)) + 0.5)
            ax.set_xticklabels([str(c) for c in keep])
            ax.set_xlim(0, len(keep))
        ax.set_ylim(pedges[-1], pedges[0])
        ax.set_yticks(pcentres)
        ax.set_yticklabels(pcodes)
        ax.tick_params(length=0, pad=6)
        ax.tick_params(axis='x', pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        start = 0
        for F, size in zip(pf_present, psizes):
            lo, hi = pcentres[start] - 0.5, pcentres[start + size - 1] + 0.5
            ax.add_patch(mpl.patches.Rectangle((-0.30, lo), 0.17, hi - lo, color=FAMILY_COLORS[F],
                                               clip_on=False, lw=0))
            start += size
        ax.set_xlabel('PC', labelpad=2)
        head(ax, L, title, dx=-26)
    if limg is not None:
        cax = ax_in(6.54, 1.43, 0.09, 1.75)
        thin_colorbar(fig, limg, cax, 'Loading', [-0.8, -0.4, 0, 0.4, 0.8])

    # Key: four family columns with colour chips.
    for i, F in enumerate(FAMILY_ORDER):
        x = 0.10 + i * 1.78
        rows = key.loc[key['family'] == F]
        fig.patches.append(mpl.patches.Rectangle((x / PAGE_WIDTH, 0.815 / fig_h), 0.09 / PAGE_WIDTH,
                                                 0.09 / fig_h, transform=fig.transFigure,
                                                 color=FAMILY_COLORS[F], lw=0))
        fig.text((x + 0.14) / PAGE_WIDTH, 0.81 / fig_h, F, fontweight='bold', va='bottom')
        for j, r in enumerate(rows.itertuples(index=False)):
            yy = (0.72 - j * 0.12) / fig_h
            fig.text(x / PAGE_WIDTH, yy, r.feature_code, fontweight='bold', color=INK2, va='top')
            fig.text((x + 0.25) / PAGE_WIDTH, yy, tidy(str(r.short_label).replace('†', '')), va='top')
    fig.add_artist(mpl.lines.Line2D([0.10 / PAGE_WIDTH, (PAGE_WIDTH - 0.10) / PAGE_WIDTH],
                                    [1.01 / fig_h] * 2, color=RULE, lw=0.5))
    return save(fig, out, 'fig3')


# ==================================================================
# Fig. 4: listener labels; B and C share one row axis.
# ==================================================================
def build_fig4(root: Path, out: Path) -> Path:
    rel = read(root, 'main_panel_A_human_reliability_context.csv')
    reg = read(root, 'feature_registry.csv', required=False)
    # Measure names as in Figs. 2-3 when the registry is available.
    short = ({} if reg is None else
             reg.set_index('feature')['short_label'].astype(str).str.replace('†', '').to_dict())
    fp = read(root, 'main_panel_B_family_correspondence.csv')
    spec = read(root, 'main_panel_C_specificity_models.csv')
    smap = read(root, 'specificity_feature_map.csv', required=False)
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER + ['QDIST', 'QTEMP'])}
    rel = rel.assign(_r=rel['signal_family'].map(fam_rank)).sort_values('_r')
    human_fams = [f for f in FAMILY_ORDER if f in set(spec['human_family'])]

    # Oriented odds ratios, as in notebook 06.
    s = fp['expected_sign']
    fp['oe'] = np.exp(s * fp['estimate'])
    fp['ol'] = np.exp(np.minimum(s * fp['ci95_low'], s * fp['ci95_high']))
    fp['oh'] = np.exp(np.maximum(s * fp['ci95_low'], s * fp['ci95_high']))

    # Shared rows: specificity feature order, then any forest-only features.
    spec_fam_col = 'feature_family' if 'feature_family' in spec else 'family'
    if smap is not None and not smap.empty:
        rows = smap[['family', 'feature', 'feature_label']].copy()
    else:
        lbl_col = 'feature_label' if 'feature_label' in spec else 'feature'
        rows = (spec[[spec_fam_col, 'feature', lbl_col]].drop_duplicates('feature')
                .rename(columns={spec_fam_col: 'family', lbl_col: 'feature_label'}))
    extra = fp.loc[~fp['feature'].isin(rows['feature']), ['family', 'feature', 'feature_label']]
    rows = pd.concat([rows, extra]).drop_duplicates('feature')
    rows = rows.assign(_r=rows['family'].map(fam_rank), _o=np.arange(len(rows))).sort_values(['_r', '_o'])
    feats = rows['feature'].tolist()
    yv = grouped_positions(rows['family'].tolist())
    ypos = dict(zip(feats, yv))
    labels = rows.set_index('feature')
    spans = {}
    for f in feats:
        spans.setdefault(labels.loc[f, 'family'], []).append(ypos[f])

    row_h = 0.20
    body_h = row_h * (yv[-1] + 1)
    top = 1.30
    fig_h = top + 0.50 + body_h + 0.50
    fig = plt.figure(figsize=(PAGE_WIDTH, fig_h))

    def ax_in(x, y, w, h):
        return fig.add_axes([x / PAGE_WIDTH, y / fig_h, w / PAGE_WIDTH, h / fig_h])

    # A: reliability as a dumbbell per signal family.
    ax = ax_in(0.62, fig_h - 1.12, 5.0, 0.88)
    xpos = np.arange(len(rel))
    for xi, a, b in zip(xpos, rel['gwet_ac1'], rel['mean_pairwise_positive_agreement']):
        ax.plot([xi, xi], [a, b], color=CONNECT, lw=1.6, solid_capstyle='butt', zorder=2)
    ax.plot(xpos, rel['gwet_ac1'], 'o', ms=4.6, color=BETWEEN_COLOR, mec='white', mew=0.5, label='Gwet’s AC1', zorder=3)
    ax.plot(xpos, rel['mean_pairwise_positive_agreement'], 's', ms=4.4, color=ACCENT, mec='white', mew=0.5,
            label='Positive agreement', zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(rel['signal_family'])
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_xlim(-0.5, len(rel) - 0.5)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0', '', '0.5', '', '1'])
    ax.set_ylabel('Agreement', labelpad=2)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), handletextpad=0.3, labelspacing=0.5,
              borderaxespad=0)
    head(ax, 'A', 'Listener agreement by signal family', dx=-40)

    # Shared labels with family rails.
    y0 = 0.50
    ylim = (yv[-1] + 0.5, -0.5)
    lab = ax_in(0.10, y0, 1.95, body_h)
    lab.set_xlim(0, 1)
    lab.set_ylim(*ylim)
    lab.axis('off')
    for F, yy in spans.items():
        family_rail(lab, F, yy[0], yy[-1], name_x=0.035)
    for f in feats:
        lab.text(0.33, ypos[f], tidy(short.get(f, labels.loc[f, 'feature_label'])),
                 ha='left', va='center')

    # B: construct-matched odds ratios.
    ax = ax_in(2.12, y0, 2.15, body_h)
    for _, r in fp.iterrows():
        if r['feature'] not in ypos:
            continue
        sig = np.isfinite(r['q_value']) and r['q_value'] < 0.05
        base = FAMILY_COLORS.get(r['family'], INK)
        col = base
        y = ypos[r['feature']]
        ax.plot([r['ol'], r['oh']], [y, y], color=col, lw=1.3, solid_capstyle='butt', zorder=3)
        ax.plot([r['oe']], [y], 'o', ms=4.8, mfc=col if sig else 'white', mec=col, mew=1.1, zorder=4)
    or_axis(ax, pd.concat([fp['ol'], fp['oh']]))
    style_rows(ax, ylim, null=1)
    ax.set_xlabel('Odds ratio (expected direction > 1)', labelpad=2)
    head(ax, 'B', 'Construct-matched association', dx=-14)

    # C: cross-family specificity, with the same family gaps as the rows.
    m = (spec.pivot_table(index='feature', columns='human_family', values='artifact_oriented_beta')
         .reindex(index=feats, columns=human_fams).to_numpy(dtype=float))
    finite = np.abs(m[np.isfinite(m)])
    lim = max(float(np.quantile(finite, 0.98)) if finite.size else 1.0, 0.25)
    yedges = []
    for yy in yv:
        yedges.append((yy - 0.5, yy + 0.5))
    cw = 0.44
    ax = ax_in(4.62, y0, cw * len(human_fams), body_h)
    norm = mpl.colors.Normalize(-lim, lim)
    for f, (lo_, hi_) in zip(feats, yedges):
        for j, hf in enumerate(human_fams):
            v = m[feats.index(f), j]
            ax.add_patch(mpl.patches.Rectangle((j, lo_), 1, 1, facecolor=DIVERGING(norm(v)) if np.isfinite(v)
                                               else 'white', edgecolor='white', lw=0.7))
        ff = labels.loc[f, 'family']
        if ff in human_fams:
            ax.add_patch(mpl.patches.Rectangle((human_fams.index(ff) + 0.04, lo_ + 0.04), 0.92, 0.92,
                                               fill=False, lw=1.1, edgecolor=INK, zorder=5))
    ax.set_xlim(0, len(human_fams))
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(len(human_fams)) + 0.5)
    ax.set_xticklabels(human_fams)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for j, hf in enumerate(human_fams):
        ax.add_patch(mpl.patches.Rectangle((j + 0.06, ylim[0] + 0.12), 0.88, 0.16, color=FAMILY_COLORS[hf],
                                           clip_on=False, lw=0))
    ax.set_xlabel('Listener-labeled family', labelpad=2)
    head(ax, 'C', 'Cross-family specificity', dx=-14)
    cax = ax_in(4.62 + cw * len(human_fams) + 0.14, y0, 0.09, body_h)
    img = mpl.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    tick = 0.5 if lim < 1.5 else 1.0
    thin_colorbar(fig, img, cax, 'Adjusted log-odds', np.arange(-np.floor(lim / tick) * tick, lim + 1e-9, tick))

    x_band0, x_band1 = 0.10 / PAGE_WIDTH, (2.12 + 2.15) / PAGE_WIDTH
    bands(fig, lab, [yy[k] for yy in spans.values() for k in range(0, len(yy), 2)], x_band0, x_band1)
    return save(fig, out, 'fig4')


def save(fig, out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in [('pdf', {}), ('svg', {}), ('png', {'dpi': 600}), ('tif', {'dpi': 600})]:
        kw2 = dict(kw)
        if ext == 'tif':
            kw2['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(out / f'{stem}.{ext}', facecolor='white', **kw2)
    plt.close(fig)
    return out / f'{stem}.pdf'


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-root', type=Path,
                        default=Path(os.environ.get('PAPER1_ANALYSIS_ROOT', repo)) / 'outputs')
    parser.add_argument('--out', type=Path, default=repo / 'outputs' / 'print_figures')
    parser.add_argument('--only', choices=['2', '3', '4'], nargs='*')
    parser.add_argument('--palette', choices=sorted(PALETTES), help='alternative colour palette')
    args = parser.parse_args()
    builders = {'2': build_fig2, '3': build_fig3, '4': build_fig4}
    if args.palette:
        apply_palette(args.palette)
    for k in args.only or ['2', '3', '4']:
        print(builders[k](args.source_root, args.out))


if __name__ == '__main__':
    main()
