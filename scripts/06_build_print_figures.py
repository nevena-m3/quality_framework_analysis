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
# ------------------------------------------------------------------
PAGE_WIDTH = 7.16
FS = 8.0          # ticks, labels, legends
FS_TITLE = 8.5    # column titles
FS_LETTER = 10.0  # panel letters

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'],
    'font.size': FS,
    'axes.titlesize': FS_TITLE,
    'axes.labelsize': FS,
    'xtick.labelsize': FS,
    'ytick.labelsize': FS,
    'legend.fontsize': FS,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

CYCLE = mpl.rcParams['axes.prop_cycle'].by_key()['color']
CONTROL_COLOR, ALS_COLOR = CYCLE[0], CYCLE[1]
BETWEEN_COLOR, WITHIN_COLOR = CYCLE[0], CYCLE[1]
DARK = '0.15'
GRID = '0.88'
FAMILY_ORDER = ['QADD', 'QGAIN', 'QREV', 'QCHAN']
FAMILY_COLORS = {f: CYCLE[i] for i, f in enumerate(FAMILY_ORDER)}
ROW_GAP = 0.65  # vertical gap between families, in row units


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


def letter(fig, ax, text, dx=-0.02, dy=0.0):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=FS_LETTER, fontweight='bold',
             ha='right', va='bottom')


def or_axis(ax, values):
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean) & (clean > 0)]
    lo, hi = (0.5, 2.0) if clean.size == 0 else (min(0.5, clean.min() / 1.25), max(2.0, clean.max() * 1.25))
    ax.set_xscale('log')
    ax.set_xlim(lo / 1.2, hi * 1.2)
    ticks = [t for t in [0.25, 0.5, 1, 2, 4, 8] if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def forest(ax, frame, positions, est, low, high, color=DARK, star=True, tier_col='inference_tier'):
    for _, row in frame.iterrows():
        if row['feature'] not in positions:
            continue
        e, lo, hi = row.get(est, np.nan), row.get(low, np.nan), row.get(high, np.nan)
        if not np.isfinite(e):
            continue
        y = positions[row['feature']]
        face = color if row.get(tier_col, '') == 'core' else 'white'
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt='o', color=color, mfc=face,
                        mec=color, ms=3.6, lw=0.9, capsize=1.6, zorder=3)
            q = row.get('q_value', np.nan)
            if star and np.isfinite(q) and q < 0.05:
                ax.text(hi, y, ' *', ha='left', va='center', fontsize=FS, color=color)
        else:
            ax.scatter(e, y, s=14, color=face, edgecolor=color, lw=0.8, zorder=3)


def style_rows(ax, ylim, separators=(), xgrid=True, null=None):
    ax.set_ylim(*ylim)
    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.spines['left'].set_visible(False)
    if xgrid:
        ax.grid(axis='x', color=GRID, lw=0.45)
    for s in separators:
        ax.axhline(s, color='0.85', lw=0.5, zorder=0)
    if null is not None:
        ax.axvline(null, color='0.55', ls='--', lw=0.7, zorder=0)


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
    for frame in [pers, long_c, long_p, coh_c, coh_p, bul_c, bul_p]:
        if 'inference_tier' not in frame:
            frame['inference_tier'] = frame['feature'].map(tier)

    # Continuous rows, grouped by family.
    pos, centers, seps, y = {}, {}, [], 0.0
    for fam in FAMILY_ORDER:
        rows = cont.loc[cont['family'] == fam]
        if rows.empty:
            continue
        if pos:
            seps.append(y - (1 + ROW_GAP) / 2)
        start = y
        for feat in rows['feature']:
            pos[feat] = y
            y += 1
        centers[fam] = (start + y - 1) / 2
        y += ROW_GAP
    ylim = (y - ROW_GAP - 1 + 0.6, -0.6)
    n_rows = y - ROW_GAP

    # Binary strip: two QCHAN deficits, then sparse QDIST/QTEMP events.
    sparse_names = ['Hard clipping', 'Dropout-like', 'Frozen audio']
    strip_rows = [(f, lbl, fam, t) for f, lbl, fam, t in
                  zip(one['feature'], one['short_label'], one['family'], one['inference_tier'])]
    strip_rows += [(f, sparse_names[i] if i < len(sparse_names) else lbl, fam, 'secondary')
                   for i, (f, lbl, fam) in enumerate(zip(spr['feature'], spr['short_label'], spr['family']))]
    spos = {r[0]: i for i, r in enumerate(strip_rows)}
    n_one = len(one)
    sylim = (len(strip_rows) - 0.4, -0.6)
    strip_seps = [n_one - 0.5] if len(spr) else []

    row_h = 0.155
    top_h, mid_h, strip_h = 0.62, 0.62, row_h * (len(strip_rows) + 0.2)
    cont_h = row_h * (n_rows + 0.2)
    fig_h = top_h + cont_h + mid_h + strip_h + 0.42
    fig = plt.figure(figsize=(PAGE_WIDTH, fig_h))
    widths = [1.62, 0.92, 0.66, 0.76, 0.72, 0.70, 0.70]
    left, right = 0.04, PAGE_WIDTH - 0.06
    gaps = (right - left - sum(widths)) / (len(widths) - 1)
    xs, x = [], left
    for w in widths:
        xs.append((x / PAGE_WIDTH, w / PAGE_WIDTH))
        x += w + gaps
    y_cont = (fig_h - top_h - cont_h) / fig_h
    y_strip = (0.42) / fig_h
    h_cont, h_strip = cont_h / fig_h, strip_h / fig_h

    def axes_pair(i):
        a = fig.add_axes([xs[i][0], y_cont, xs[i][1], h_cont])
        b = fig.add_axes([xs[i][0], y_strip, xs[i][1], h_strip])
        return a, b

    # Labels column.
    lab, slab = axes_pair(0)
    for ax, lim in [(lab, ylim), (slab, sylim)]:
        ax.set_xlim(0, 1)
        ax.set_ylim(*lim)
        ax.axis('off')
    for fam, c in centers.items():
        lab.text(0.0, c, fam, ha='left', va='center', fontweight='bold', color='0.40')
    for r in cont.itertuples(index=False):
        lab.text(0.27, pos[r.feature], r.short_label, ha='left', va='center',
                 fontweight='bold' if r.inference_tier == 'core' else 'normal')
    fam_strip = {}
    for f, lbl, fam, t in strip_rows:
        fam_strip.setdefault(fam, []).append(spos[f])
        slab.text(0.27, spos[f], lbl, ha='left', va='center',
                  fontweight='bold' if t == 'core' else 'normal')
    for fam, ys in fam_strip.items():
        slab.text(0.0, np.mean(ys), fam, ha='left', va='center', fontweight='bold', color='0.40')

    # A: distributions (continuous) and prevalence (strip).
    ax, sx = axes_pair(1)
    for r in cont.itertuples(index=False):
        for cohort, off, col in [('CONTROL', -0.17, CONTROL_COLOR), ('ALS', 0.17, ALS_COLOR)]:
            s = std.loc[(std['feature'] == r.feature) & (std['cohort'] == cohort)]
            if s.empty:
                continue
            s = s.iloc[0]
            yy = pos[r.feature] + off
            ax.plot([s['p05_z'], s['p95_z']], [yy, yy], color=col, lw=0.7, alpha=0.8)
            ax.plot([s['q1_z'], s['q3_z']], [yy, yy], color=col, lw=2.0, solid_capstyle='round')
            ax.scatter(s['median_z'], yy, s=11, color=col, edgecolor='white', lw=0.4, zorder=3)
    style_rows(ax, ylim, seps, null=0)
    ax.set_xlim(-3.2, 3.2)
    ax.set_xticks([-3, 0, 3])
    ax.set_xlabel('Transformed value\n(IQR units)', labelpad=2)
    ax.set_title(' ', loc='left', pad=4)
    lab.legend(handles=[
        mpl.lines.Line2D([0], [0], marker='o', ms=3.5, color=CONTROL_COLOR, lw=1.5, label='Control'),
        mpl.lines.Line2D([0], [0], marker='o', ms=3.5, color=ALS_COLOR, lw=1.5, label='ALS')],
        frameon=False, ncol=1, loc='lower left', bbox_to_anchor=(0.0, 1.0),
        handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    smax = 0.0
    for f, *_ in strip_rows:
        for cohort, off, col, mk in [('CONTROL', -0.17, CONTROL_COLOR, 'o'), ('ALS', 0.17, ALS_COLOR, 's')]:
            if f in set(one['feature']):
                rec = states.loc[(states['feature'] == f) & (states['cohort'] == cohort)]
                val = 100 - float(rec.iloc[0]['zero_percent_available']) if not rec.empty else np.nan
            else:
                rec = sparse.loc[(sparse['feature'] == f) & (sparse['cohort'] == cohort)]
                val = (100 * float(rec.iloc[0]['positive_count']) / float(rec.iloc[0]['n_available_recordings'])
                       if not rec.empty and float(rec.iloc[0]['n_available_recordings']) > 0 else np.nan)
            if np.isfinite(val):
                smax = max(smax, val)
                sx.scatter(val, spos[f] + off, marker=mk, s=13, color=col, edgecolor='white', lw=0.3, zorder=3)
    style_rows(sx, sylim, strip_seps)
    sx.set_xlim(-8, 108)
    sx.set_xticks([0, 50, 100])
    sx.set_xlabel('Recordings (%)', labelpad=2)

    # B: persistence and annual change.
    ax, sx = axes_pair(2)
    icc = pers.rename(columns={'icc': 'e', 'icc_ci95_low': 'l', 'icc_ci95_high': 'h'})
    forest(ax, icc, pos, 'e', 'l', 'h', star=False)
    forest(sx, icc, spos, 'e', 'l', 'h', star=False)
    for a, lim, sp in [(ax, ylim, seps), (sx, sylim, strip_seps)]:
        style_rows(a, lim, sp)
        a.set_xlim(-0.08, 1.08)
        a.set_xticks([0, 0.5, 1])
        a.set_xticklabels(['0', '0.5', '1'])
    ax.set_title('Persistence', loc='left', pad=4)
    sx.set_xlabel('ICC', labelpad=2)
    ax.set_xlabel('ICC', labelpad=2)

    ax, sx = axes_pair(3)
    forest(ax, long_c, pos, 'estimate', 'ci95_low', 'ci95_high')
    style_rows(ax, ylim, seps, null=0)
    ax.set_title('Annual change', loc='left', pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('IQR units\nper year', labelpad=2)
    lp = long_p.copy()
    if not lp.empty:
        lp['o'], lp['ol'], lp['oh'] = np.exp(lp['estimate']), np.exp(lp['ci95_low']), np.exp(lp['ci95_high'])
        forest(sx, lp, spos, 'o', 'ol', 'oh')
        or_axis(sx, pd.concat([lp['ol'], lp['oh']]))
    style_rows(sx, sylim, strip_seps, null=1)
    sx.set_xlabel('Annual OR', labelpad=2)

    # C: cohort difference and bulbar associations (per point of worse bulbar function).
    ax, sx = axes_pair(4)
    forest(ax, coh_c, pos, 'estimate', 'ci95_low', 'ci95_high')
    style_rows(ax, ylim, seps, null=0)
    ax.set_title('ALS − control', loc='left', pad=4)
    ax.set_xlabel('IQR units', labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    cp = coh_p.copy()
    cp['o'], cp['ol'], cp['oh'] = np.exp(cp['estimate']), np.exp(cp['ci95_low']), np.exp(cp['ci95_high'])
    forest(sx, cp, spos, 'o', 'ol', 'oh')
    or_axis(sx, pd.concat([cp['ol'], cp['oh']]))
    style_rows(sx, sylim, strip_seps, null=1)
    sx.set_xlabel('Odds ratio', labelpad=2)

    for i, term, title, col in [(5, 'bulbar_between', 'Bulbar, between', BETWEEN_COLOR),
                                (6, 'bulbar_within', 'Bulbar, within', WITHIN_COLOR)]:  # noqa: E501
        ax, sx = axes_pair(i)
        f = bul_c.loc[bul_c['target_term'].eq(term)].copy()
        f['d'], f['dl'], f['dh'] = -f['estimate'], -f['ci95_high'], -f['ci95_low']
        forest(ax, f, pos, 'd', 'dl', 'dh', color=col)
        style_rows(ax, ylim, seps, null=0)
        ax.set_title(title.split(', ')[1].capitalize(), loc='left', pad=4)
        ax.set_xlabel('IQR units per\npoint worse', labelpad=2)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        p = bul_p.loc[bul_p['target_term'].eq(term)].copy()
        p['o'], p['ol'], p['oh'] = np.exp(-p['estimate']), np.exp(-p['ci95_high']), np.exp(-p['ci95_low'])
        forest(sx, p, spos, 'o', 'ol', 'oh', color=col)
        or_axis(sx, pd.concat([p['ol'], p['oh']]))
        style_rows(sx, sylim, strip_seps, null=1)
        sx.set_xlabel('OR per point', labelpad=2)

    # Group headers spanning their columns.
    y_hdr = (fig_h - 0.20) / fig_h
    for i, text in [(1, 'A  Distribution'), (2, 'B  Persistence and change'),
                    (4, 'C  Cohort and bulbar associations')]:
        fig.text(xs[i][0], y_hdr, text, fontsize=FS_TITLE, fontweight='bold', ha='left', va='bottom')
    for i, j in [(2, 3), (4, 6)]:
        x0, x1 = xs[i][0], xs[j][0] + xs[j][1]
        fig.add_artist(mpl.lines.Line2D([x0, x1], [y_hdr - 0.03 / fig_h] * 2, color='0.6', lw=0.6))
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

    fig_h = 6.80
    fig = plt.figure(figsize=(PAGE_WIDTH, fig_h))

    def ax_in(x, y, w, h):  # inches from bottom-left
        return fig.add_axes([x / PAGE_WIDTH, y / fig_h, w / PAGE_WIDTH, h / fig_h])

    cmap = mpl.colormaps['RdBu_r'].copy()
    cmap.set_bad('#e6e6e6')
    counts = [sum(1 for f in order if fam[f] == F) for F in FAMILY_ORDER]
    bounds = np.cumsum(counts)[:-1] - 0.5
    hm = 2.55
    img = None
    for x0, col, L, title in [(0.42, 'between_rho', 'A', 'Between participants'),
                              (3.55, 'within_rho', 'B', 'Within participants')]:
        ax = ax_in(x0, 3.80, hm, hm)
        m = matrix(col)
        img = ax.imshow(np.ma.masked_invalid(m), vmin=-1, vmax=1, cmap=cmap, aspect='equal')
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels([codes[f] for f in order], rotation=90, fontsize=FS)
        ax.set_yticklabels([codes[f] for f in order], fontsize=FS)
        for lbls in (ax.get_xticklabels(), ax.get_yticklabels()):
            for t, f in zip(lbls, order):
                t.set_color(FAMILY_COLORS[fam[f]])
                t.set_fontweight('bold')
        for b in bounds:
            ax.axhline(b, color='white', lw=1.1)
            ax.axvline(b, color='white', lw=1.1)
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc='left', fontweight='bold', pad=4)
        fig.text((x0 - 0.30) / PAGE_WIDTH, (3.80 + hm + 0.06) / fig_h, L, fontsize=FS_LETTER,
                 fontweight='bold', ha='left', va='bottom')
    cax = ax_in(6.35, 4.10, 0.10, 1.95)
    cb = fig.colorbar(img, cax=cax)
    cb.set_label('Rank correlation', labelpad=3)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    # C: between vs within scatter.
    ax = ax_in(0.55, 1.30, 1.75, 1.75)
    d = pairs.loc[pairs['between_rho'].notna() & pairs['within_rho'].notna()]
    bf = d.loc[~d['same_family'] & ~d['structural_dependency']]
    wf = d.loc[d['same_family'] & ~d['structural_dependency']]
    sd = d.loc[d['structural_dependency']]
    for v in (0,):
        ax.axhline(v, color='0.8', lw=0.6)
        ax.axvline(v, color='0.8', lw=0.6)
    ax.plot([-1, 1], [-1, 1], ls='--', color='0.55', lw=0.7)
    ax.scatter(bf['between_rho'], bf['within_rho'], s=10, facecolors='none', edgecolors='0.55', lw=0.6,
               label='Between-family')
    ax.scatter(wf['between_rho'], wf['within_rho'], s=13, color=CYCLE[0], alpha=0.8, lw=0, label='Within-family')
    ax.scatter(sd['between_rho'], sd['within_rho'], s=22, marker='D', color=CYCLE[1], edgecolor='white',
               lw=0.4, label='Mechanically related', zorder=4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel('Between-participant ρ', labelpad=2)
    ax.set_ylabel('Within-participant ρ', labelpad=1)
    ax.legend(frameon=False, loc='upper left', ncol=1, handletextpad=0.2, borderaxespad=0.1,
              labelspacing=0.25)
    fig.text(0.05 / PAGE_WIDTH, 3.13 / fig_h, 'C', fontsize=FS_LETTER, fontweight='bold', va='bottom')

    # D: scree with parallel analysis.
    ax = ax_in(2.85, 1.30, 1.45, 1.75)
    n = comps.groupby('level')['component'].max().max()
    for level, mk, col in [('between', 'o', CYCLE[0]), ('within', 's', CYCLE[1])]:
        c = comps.loc[comps['level'] == level].sort_values('component')
        ax.plot(c['component'], c['variance_explained_percent'], marker=mk, ms=3.2, lw=1.0, color=col,
                label=level.capitalize())
        ax.plot(c['component'], 100 * c['parallel_95th_eigenvalue'] / n, ls='--' if level == 'between' else ':',
                lw=0.9, color=col, label=f'{level.capitalize()}, 95th PA')
    lo_v, hi_v = ax.get_ylim()
    ax.set_ylim(lo_v, hi_v + 0.55 * (hi_v - lo_v))
    ax.set_xticks(range(1, int(n) + 1))
    ax.set_xlabel('Component', labelpad=2)
    ax.set_ylabel('Variance explained (%)', labelpad=2)
    ax.grid(axis='y', color=GRID, lw=0.45)
    ax.legend(frameon=False, loc='upper right', ncol=1, handlelength=1.6, borderaxespad=0.1,
              labelspacing=0.25)
    fig.text(2.38 / PAGE_WIDTH, 3.13 / fig_h, 'D', fontsize=FS_LETTER, fontweight='bold', va='bottom')

    # E, F: retained loadings.
    pca_feats = (loads.drop_duplicates('feature').assign(_r=lambda f: f['feature'].map(
        {ft: i for i, ft in enumerate(order)})).sort_values('_r'))
    pcodes = pca_feats['feature_code'].tolist()
    lcmap = mpl.colormaps['RdBu_r'].copy()
    lcmap.set_bad('#e6e6e6')
    limg = None
    for x0, level, L, title in [(4.78, 'between', 'E', 'Between'), (5.72, 'within', 'F', 'Within')]:
        keep = comps.loc[(comps['level'] == level) & comps['retained_stable_component'], 'component'].tolist()
        ax = ax_in(x0, 1.30, 0.2 * max(len(keep), 1), 1.75)
        if keep:
            piv = (loads.loc[(loads['level'] == level) & loads['component'].isin(keep)]
                   .pivot(index='feature_code', columns='component', values='loading').reindex(pcodes))
            limg = ax.imshow(np.ma.masked_invalid(piv.to_numpy(dtype=float)), vmin=-0.8, vmax=0.8,
                             cmap=lcmap, aspect='auto')
            ax.set_xticks(range(len(keep)))
            ax.set_xticklabels([str(k) for k in keep])
        ax.set_yticks(range(len(pcodes)))
        ax.set_yticklabels(pcodes, fontsize=FS)
        for t, f in zip(ax.get_yticklabels(), pca_feats['feature']):
            t.set_color(FAMILY_COLORS[fam.get(f, 'QADD')])
            t.set_fontweight('bold')
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel('PC', labelpad=2)
        ax.set_title(title, loc='left', fontweight='bold', pad=4)
        fig.text((x0 - 0.34) / PAGE_WIDTH, 3.13 / fig_h, L, fontsize=FS_LETTER, fontweight='bold', va='bottom')
    if limg is not None:
        cax = ax_in(6.62, 1.30, 0.08, 1.75)
        cb = fig.colorbar(limg, cax=cax)
        cb.set_ticks([-0.8, 0, 0.8])
        cb.set_label('Loading', labelpad=1)

    # Key: four family columns.
    for i, F in enumerate(FAMILY_ORDER):
        x = (0.05 + i * 1.78) / PAGE_WIDTH
        rows = key.loc[key['family'] == F]
        lines = [f"{r.feature_code}  {str(r.short_label).replace('†', '')}" for r in rows.itertuples(index=False)]
        fig.text(x, 0.66 / fig_h, F, color=FAMILY_COLORS[F], fontweight='bold', va='bottom')
        fig.text(x, 0.62 / fig_h, '\n'.join(lines), va='top', linespacing=1.15)
    return save(fig, out, 'fig3')


# ==================================================================
# Fig. 4: listener labels; B and C share one row axis.
# ==================================================================
def build_fig4(root: Path, out: Path) -> Path:
    rel = read(root, 'main_panel_A_human_reliability_context.csv')
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
    if smap is not None:
        rows = smap[['family', 'feature', 'feature_label']].copy()
    else:
        lbl_col = 'feature_label' if 'feature_label' in spec else 'feature'
        rows = (spec[[spec_fam_col, 'feature', lbl_col]].drop_duplicates('feature')
                .rename(columns={spec_fam_col: 'family', lbl_col: 'feature_label'}))
    extra = fp.loc[~fp['feature'].isin(rows['feature']), ['family', 'feature', 'feature_label']]
    rows = pd.concat([rows, extra]).drop_duplicates('feature')
    rows = rows.assign(_r=rows['family'].map(fam_rank), _o=np.arange(len(rows))).sort_values(['_r', '_o'])
    feats = rows['feature'].tolist()
    ypos = {f: i for i, f in enumerate(feats)}
    seps = [i - 0.5 for i in range(1, len(rows)) if rows['family'].iloc[i] != rows['family'].iloc[i - 1]]

    row_h = 0.205
    body_h = row_h * len(feats)
    fig_h = 1.55 + 0.45 + body_h + 0.55
    fig = plt.figure(figsize=(PAGE_WIDTH, fig_h))

    def ax_in(x, y, w, h):
        return fig.add_axes([x / PAGE_WIDTH, y / fig_h, w / PAGE_WIDTH, h / fig_h])

    # A: reliability.
    ax = ax_in(0.55, fig_h - 1.32, 4.1, 1.12)
    xpos = np.arange(len(rel))
    ax.scatter(xpos - 0.1, rel['gwet_ac1'], marker='o', s=18, color=CYCLE[0], label='Gwet AC1', zorder=3)
    ax.scatter(xpos + 0.1, rel['mean_pairwise_positive_agreement'], marker='s', s=18, color=CYCLE[1],
               label='Positive agreement', zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(rel['signal_family'])
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('Agreement', labelpad=2)
    ax.grid(axis='y', color=GRID, lw=0.45)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.03, 0.5), handletextpad=0.3)
    fig.text(0.05 / PAGE_WIDTH, (fig_h - 0.22) / fig_h, 'A', fontsize=FS_LETTER, fontweight='bold', va='bottom')

    # Shared labels.
    y0 = 0.55
    ylim = (len(feats) - 0.5, -0.5)
    lab = ax_in(0.05, y0, 1.95, body_h)
    lab.set_xlim(0, 1)
    lab.set_ylim(*ylim)
    lab.axis('off')
    labels = rows.set_index('feature')
    for f in feats:
        lab.text(0.0, ypos[f], labels.loc[f, 'family'], ha='left', va='center', fontweight='bold',
                 color=FAMILY_COLORS.get(labels.loc[f, 'family'], '0.3'))
        lab.text(0.30, ypos[f], str(labels.loc[f, 'feature_label']).replace(' present', ''),
                 ha='left', va='center')

    # B: construct-matched odds ratios.
    ax = ax_in(2.05, y0, 2.1, body_h)
    for _, r in fp.iterrows():
        if r['feature'] not in ypos:
            continue
        col = FAMILY_COLORS.get(r['family'], DARK)
        sig = np.isfinite(r['q_value']) and r['q_value'] < 0.05
        ax.errorbar(r['oe'], ypos[r['feature']], xerr=[[r['oe'] - r['ol']], [r['oh'] - r['oe']]], fmt='o',
                    color=col, mec=col, mfc=col if sig else 'white', ms=4, lw=1.0, capsize=1.8, zorder=3)
    or_axis(ax, pd.concat([fp['ol'], fp['oh']]))
    style_rows(ax, ylim, seps, null=1)
    ax.set_xlabel('Odds ratio (>1 = prespecified direction)', labelpad=2)
    ax.set_title('Construct-matched association', loc='left', fontweight='bold', pad=4)
    fig.text(1.98 / PAGE_WIDTH, (y0 + body_h + 0.05) / fig_h, 'B', fontsize=FS_LETTER, fontweight='bold',
             va='bottom', ha='right')

    # C: cross-family specificity.
    m = (spec.pivot_table(index='feature', columns='human_family', values='artifact_oriented_beta')
         .reindex(index=feats, columns=human_fams).to_numpy(dtype=float))
    finite = np.abs(m[np.isfinite(m)])
    lim = max(float(np.quantile(finite, 0.98)) if finite.size else 1.0, 0.25)
    cmap = mpl.colormaps['RdBu_r'].copy()
    cmap.set_bad('#e6e6e6')
    ax = ax_in(4.55, y0, 0.42 * len(human_fams), body_h)
    img = ax.imshow(np.ma.masked_invalid(m), aspect='auto', cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(human_fams)))
    ax.set_xticklabels(human_fams, rotation=0, fontsize=FS)
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for f in feats:
        ff = labels.loc[f, 'family']
        if ff in human_fams:
            ax.add_patch(mpl.patches.Rectangle((human_fams.index(ff) - 0.5, ypos[f] - 0.5), 1, 1, fill=False,
                                               lw=1.2, edgecolor='black'))
    ax.set_xlabel('Listener family', labelpad=2)
    ax.set_title('Cross-family', loc='left', fontweight='bold', pad=4)
    fig.text(4.50 / PAGE_WIDTH, (y0 + body_h + 0.05) / fig_h, 'C', fontsize=FS_LETTER, fontweight='bold',
             va='bottom', ha='right')
    cax = ax_in(4.55 + 0.42 * len(human_fams) + 0.1, y0, 0.09, body_h)
    cb = fig.colorbar(img, cax=cax)
    cb.set_label('Adjusted log-odds', labelpad=3)
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
    args = parser.parse_args()
    builders = {'2': build_fig2, '3': build_fig3, '4': build_fig4}
    for k in args.only or ['2', '3', '4']:
        print(builders[k](args.source_root, args.out))


if __name__ == '__main__':
    main()
