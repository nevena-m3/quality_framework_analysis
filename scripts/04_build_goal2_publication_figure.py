from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

# Resolve project/output root from notebook globals when available.
if 'OUTPUT_ROOT' in globals():
    output_root = Path(OUTPUT_ROOT)
else:
    output_root = Path(r'C:\Users\musikicn\Desktop\Nevena_project\Paper_1\analysis_recording_quality\outputs\goal2_cohort_characterization')
    if not output_root.exists():
        output_root = Path('/tmp/goal2review')

# Support the extracted review ZIP used during testing, whose filenames include backslashes.
def p(rel):
    direct = output_root / rel
    if direct.exists():
        return direct
    escaped = output_root / ('goal2_outputs\\' + rel.replace('/', '\\'))
    if escaped.exists():
        return escaped
    raise FileNotFoundError(f'Could not find {rel} below {output_root}')

availability = pd.read_csv(p('tables/goal2_availability_summary.csv'))
registry = pd.read_csv(p('tables/goal2_feature_registry.csv'))
cohort_models = pd.read_csv(p('tables/goal2_cohort_models.csv'))
bulbar_models = pd.read_csv(p('tables/goal2_bulbar_models.csv'))
distributions = pd.read_csv(p('figures/source_data/goal2_panel_B_distributions.csv'))
sparse_gates = pd.read_csv(p('tables/goal2_sparse_event_model_gates.csv'))

fig_dir = output_root / 'figures' / 'main'
src_dir = output_root / 'figures' / 'source_data'
fig_dir.mkdir(parents=True, exist_ok=True)
src_dir.mkdir(parents=True, exist_ok=True)

# Main manuscript indicators, deliberately excluding sparse QDIST from inferential panels.
main_features = [
    'qadd_pause_ac_level_dbfs_median',
    'qgain_typical_speech_level_dbfs',
    'qgain_within_segment_iqr_db',
    'qrev_tail_excess_100ms_db',
    'qrev_srmr_norm',
    'qchan_ltas_distance_db',
    'qchan_rolloff95_deficit_hz',
    'qdist_hard_clipped_sample_fraction',
]
label_map = registry.set_index('feature')['label'].to_dict()
family_map = registry.set_index('feature')['family'].to_dict()

# Nature-like restrained palette.
COL_CONTROL = '#3B6EA8'
COL_ALS = '#D89000'
COL_BETWEEN = '#2C6E9B'
COL_WITHIN = '#B45A87'
COL_TEXT = '#202124'
COL_MUTED = '#6B7280'
COL_GRID = '#D9DDE3'
COL_SIG = '#111111'
COL_NS = '#FFFFFF'
FAMILY_COLORS = {
    'QADD': '#5E81AC',
    'QGAIN': '#D08770',
    'QREV': '#A3BE8C',
    'QCHAN': '#B48EAD',
    'QDIST': '#88C0D0',
}

mpl.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9.2,
    'axes.labelsize': 8.2,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
    'legend.fontsize': 7.2,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

fig = plt.figure(figsize=(7.2, 11.2), facecolor='white')
outer = GridSpec(4, 1, figure=fig, height_ratios=[1.05, 1.72, 1.18, 1.25], hspace=0.62)
axA = fig.add_subplot(outer[0, 0])

# ---------------- Panel A: availability matrix ----------------
a_feats = main_features
rows = []
for feature in a_feats:
    for cohort in ['CONTROL', 'ALS']:
        sub = availability[(availability.feature == feature) & (availability.cohort == cohort)]
        if not sub.empty:
            rows.append(sub.iloc[0].to_dict())
panelA = pd.DataFrame(rows)
panelA.to_csv(src_dir / 'goal2_publication_panel_A.csv', index=False)

arr = np.full((len(a_feats), 2), np.nan)
for i, feature in enumerate(a_feats):
    for j, cohort in enumerate(['CONTROL', 'ALS']):
        sub = panelA[(panelA.feature == feature) & (panelA.cohort == cohort)]
        if not sub.empty:
            arr[i, j] = float(sub.availability_percent.iloc[0])

# Heatmap cells, manually rendered for precise print control.
axA.set_xlim(-0.55, 3.55)
axA.set_ylim(len(a_feats)-0.5, -0.5)
for i, feature in enumerate(a_feats):
    fam = family_map.get(feature, '')
    axA.add_patch(Rectangle((-0.52, i-0.38), 0.055, 0.76, facecolor=FAMILY_COLORS.get(fam, COL_MUTED), edgecolor='none'))
    for j in range(2):
        v = arr[i, j]
        shade = 0.96 - 0.55*(v/100.0) if np.isfinite(v) else 0.96
        # neutral blue scale to avoid conflating cohort color with completeness
        color = mpl.colors.to_rgba('#2F6B9A', alpha=0.15 + 0.65*(v/100.0)) if np.isfinite(v) else (0.95,0.95,0.95,1)
        axA.add_patch(Rectangle((j-0.42, i-0.34), 0.84, 0.68, facecolor=color, edgecolor='white', linewidth=1.1))
        axA.text(j, i, f'{v:.0f}%', ha='center', va='center', color='white' if v >= 72 else COL_TEXT, fontweight='bold' if v < 50 else 'normal', fontsize=7.5)

    # Right-side support annotation, emphasizing conditional and sparse features.
    allrow = availability[(availability.feature == feature) & (availability.cohort == 'ALL')]
    if not allrow.empty:
        r = allrow.iloc[0]
        support = f"{int(r.n_available_recordings)}/{int(r.n_recordings)}"
        if feature == 'qdist_hard_clipped_sample_fraction':
            support += f"; {int(r.n_positive)} positive"
        elif feature == 'qchan_rolloff95_deficit_hz':
            support += f"; {int(r.n_positive)} deficit-positive"
        elif feature == 'qrev_tail_excess_100ms_db':
            support += '; conditional support'
        axA.text(2.0, i, support, ha='left', va='center', color=COL_MUTED, fontsize=6.8)

labelsA = [label_map[f] + ('†' if f == 'qrev_tail_excess_100ms_db' else '') for f in a_feats]
axA.set_yticks(np.arange(len(a_feats)))
axA.set_yticklabels(labelsA)
axA.set_xticks([0,1])
axA.set_xticklabels(['Control', 'ALS'])
axA.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
axA.tick_params(axis='y', length=0)
for s in axA.spines.values(): s.set_visible(False)
axA.set_title('A   Measurement coverage across the analysis population', loc='left', fontweight='bold', pad=14)
axA.text(2.0, -0.62, 'Available recordings / total', ha='left', va='bottom', fontsize=7.1, color=COL_TEXT)
axA.text(3.52, len(a_feats)-0.15, '† Boundary-dependent measurement', ha='right', va='top', fontsize=6.5, color=COL_MUTED)

# ---------------- Panel B: small-multiple distributions ----------------
subB = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1, 0], hspace=0.68, wspace=0.44)
b_features = [
    'qadd_pause_ac_level_dbfs_median',
    'qgain_typical_speech_level_dbfs',
    'qgain_within_segment_iqr_db',
    'qrev_tail_excess_100ms_db',
    'qrev_srmr_norm',
    'qchan_ltas_distance_db',
]
panelB = distributions[distributions.feature.isin(b_features)].copy()
panelB.to_csv(src_dir / 'goal2_publication_panel_B.csv', index=False)

for k, feature in enumerate(b_features):
    ax = fig.add_subplot(subB[k//3, k%3])
    for xpos, cohort, color in [(0, 'CONTROL', COL_CONTROL), (1, 'ALS', COL_ALS)]:
        vals = panelB[(panelB.feature == feature) & (panelB.cohort == cohort)].standardized_value.dropna().to_numpy()
        if vals.size >= 3 and np.nanstd(vals) > 0:
            lo, hi = np.nanpercentile(vals, [1,99])
            grid = np.linspace(lo, hi, 180)
            try:
                density = gaussian_kde(vals)(grid)
                density = density / density.max() * 0.34
                ax.fill_betweenx(grid, xpos-density, xpos+density, color=color, alpha=0.28, linewidth=0)
                ax.plot(xpos+density, grid, color=color, lw=0.7, alpha=0.85)
                ax.plot(xpos-density, grid, color=color, lw=0.7, alpha=0.85)
            except Exception:
                pass
        q1, med, q3 = np.nanpercentile(vals, [25,50,75])
        low, high = np.nanpercentile(vals, [5,95])
        ax.plot([xpos, xpos], [low, high], color=color, lw=0.8)
        ax.plot([xpos-0.12, xpos+0.12], [q1,q1], color=color, lw=1.1)
        ax.plot([xpos-0.12, xpos+0.12], [q3,q3], color=color, lw=1.1)
        ax.add_patch(Rectangle((xpos-0.12,q1),0.24,q3-q1,facecolor='white',edgecolor=color,lw=0.9,zorder=3))
        ax.plot([xpos-0.13,xpos+0.13],[med,med],color=COL_TEXT,lw=1.1,zorder=4)
        ax.text(xpos, ax.get_ylim()[1] if False else np.nanpercentile(vals,99), f'n={len(vals)}', ha='center', va='bottom', fontsize=5.8, color=COL_MUTED)
    ax.axhline(0, color=COL_MUTED, lw=0.65, ls=(0,(3,2)), alpha=0.7)
    ax.set_xlim(-0.55,1.55)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Control','ALS'])
    ax.set_title(label_map[feature] + ('†' if feature=='qrev_tail_excess_100ms_db' else ''), loc='left', fontsize=7.7, fontweight='bold', pad=4)
    ax.grid(axis='y', color=COL_GRID, lw=0.45, alpha=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylabel('')

# Panel B section title using figure coordinates.
fig.text(0.078, 0.585, 'Standardized value (median-centred, per IQR)', rotation=90, va='center', ha='center', fontsize=8.2, color=COL_TEXT)
fig.text(0.105, 0.735, 'B   Cohort distributions of the principal continuous indicators', fontweight='bold', fontsize=9.2)

# ---------------- Panel C: cohort effects, continuous + presence ----------------
subC = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, 0], width_ratios=[1.62, 0.92], wspace=0.52)
axC1 = fig.add_subplot(subC[0,0])
axC2 = fig.add_subplot(subC[0,1])

c_cont_features = b_features
c1 = cohort_models[(cohort_models.analysis == 'cohort_continuous') & cohort_models.feature.isin(c_cont_features)].copy()
c1['order'] = c1.feature.map({f:i for i,f in enumerate(c_cont_features)})
c1 = c1.sort_values('order', ascending=False)
c1.to_csv(src_dir / 'goal2_publication_panel_C_continuous.csv', index=False)
y = np.arange(len(c1))
for yi, r in zip(y, c1.itertuples()):
    sig = np.isfinite(r.q_value) and r.q_value < 0.05
    axC1.errorbar(r.estimate, yi, xerr=[[r.estimate-r.ci95_low],[r.ci95_high-r.estimate]], fmt='o', mfc=COL_SIG if sig else 'white', mec='#1C6E9C', mew=1.0, color='#1C6E9C', ecolor='#1C6E9C', capsize=2.2, ms=4.6, lw=1.0)
axC1.axvline(0,color=COL_MUTED,lw=0.7,ls=(0,(3,2)))
axC1.set_yticks(y); axC1.set_yticklabels([label_map[f] + ('†' if f=='qrev_tail_excess_100ms_db' else '') for f in c1.feature])
axC1.set_xlabel('ALS − control\n(standardized difference)')
axC1.grid(axis='x', color=COL_GRID,lw=0.45,alpha=0.8)
axC1.spines['top'].set_visible(False); axC1.spines['right'].set_visible(False)
axC1.set_title('Continuous indicators', loc='left', fontsize=7.5, fontweight='bold')

c2_features = ['qchan_rolloff95_deficit_hz','qchan_highband_ratio_deficit']
c2 = cohort_models[(cohort_models.analysis == 'cohort_presence') & cohort_models.feature.isin(c2_features)].copy()
c2['OR'] = np.exp(c2.estimate); c2['lo'] = np.exp(c2.ci95_low); c2['hi'] = np.exp(c2.ci95_high)
c2['order'] = c2.feature.map({f:i for i,f in enumerate(c2_features)})
c2 = c2.sort_values('order', ascending=False)
c2.to_csv(src_dir / 'goal2_publication_panel_C_presence.csv', index=False)
y2=np.arange(len(c2))
for yi,r in zip(y2,c2.itertuples()):
    sig=np.isfinite(r.q_value) and r.q_value<0.05
    axC2.errorbar(r.OR, yi, xerr=[[r.OR-r.lo],[r.hi-r.OR]], fmt='o', mfc=COL_SIG if sig else 'white', mec='#8A5A9B', mew=1.0, color='#8A5A9B', ecolor='#8A5A9B', capsize=2.2, ms=4.6, lw=1.0)
axC2.axvline(1,color=COL_MUTED,lw=0.7,ls=(0,(3,2)))
axC2.set_xscale('log')
axC2.set_xticks([1,2,4])
axC2.set_xticklabels(['1','2','4'])
axC2.set_yticks(y2); axC2.set_yticklabels(['Rolloff-95 deficit','High-band deficit'][:len(c2)][::-1] if len(c2)==2 else [label_map[f] for f in c2.feature])
axC2.set_xlabel('ALS/control\nodds ratio')
axC2.grid(axis='x', color=COL_GRID,lw=0.45,alpha=0.8)
axC2.spines['top'].set_visible(False); axC2.spines['right'].set_visible(False)
axC2.set_title('Directional-deficit presence', loc='left', fontsize=7.5, fontweight='bold')

# ---------------- Panel D: bulbar effects, sign reoriented ----------------
axD = fig.add_subplot(outer[3,0])
d_features = b_features
D = bulbar_models[(bulbar_models.analysis.isin(['bulbar_between','bulbar_within'])) & bulbar_models.feature.isin(d_features)].copy()
D['display_estimate'] = -D.estimate
D['display_low'] = -D.ci95_high
D['display_high'] = -D.ci95_low
D['order'] = D.feature.map({f:i for i,f in enumerate(d_features)})
D.to_csv(src_dir / 'goal2_publication_panel_D.csv', index=False)
orderD = [f for f in d_features if f in set(D.feature)][::-1]
yD=np.arange(len(orderD))
for analysis_name,offset,color,label in [('bulbar_between',-0.11,COL_BETWEEN,'Between participant'),('bulbar_within',0.11,COL_WITHIN,'Within participant')]:
    sub=D[D.analysis==analysis_name].set_index('feature').reindex(orderD)
    for yi,(feature,r) in enumerate(sub.iterrows()):
        if pd.isna(r.display_estimate): continue
        sig=np.isfinite(r.q_value) and r.q_value<0.05
        axD.errorbar(r.display_estimate, yi+offset, xerr=[[r.display_estimate-r.display_low],[r.display_high-r.display_estimate]], fmt='o', mfc=color if sig else 'white', mec=color, mew=1.0, color=color, ecolor=color, capsize=2.2, ms=4.4, lw=1.0)
axD.axvline(0,color=COL_MUTED,lw=0.7,ls=(0,(3,2)))
axD.set_yticks(yD); axD.set_yticklabels([label_map[f] + ('†' if f=='qrev_tail_excess_100ms_db' else '') for f in orderD])
axD.set_xlabel('Effect per one-point lower bulbar score\n(worse bulbar function; standardized units)')
axD.grid(axis='x',color=COL_GRID,lw=0.45,alpha=0.8)
axD.spines['top'].set_visible(False); axD.spines['right'].set_visible(False)
handles=[mpl.lines.Line2D([],[],marker='o',color=COL_BETWEEN,mfc=COL_BETWEEN,lw=0,label='Between participant'),mpl.lines.Line2D([],[],marker='o',color=COL_WITHIN,mfc=COL_WITHIN,lw=0,label='Within participant')]
axD.legend(handles=handles, frameon=False, loc='upper right', handletextpad=0.4, ncol=2)

# Titles and interpretive aids.
fig.text(0.105, 0.452, 'C   Cohort differences', fontweight='bold', fontsize=9.2)
fig.text(0.105, 0.238, 'D   Associations with bulbar function in ALS', fontweight='bold', fontsize=9.2)
fig.suptitle('Support, distributions, and clinical-group structure of recording-quality indicators', x=0.105, ha='left', y=0.988, fontsize=12.0, fontweight='bold', color=COL_TEXT)
fig.text(0.105,0.965,'Main indicators are shown without collapsing families into a global quality score.',ha='left',fontsize=7.5,color=COL_MUTED)
fig.text(0.105,0.014,'Filled symbols indicate BH-FDR q < 0.05. QDIST was descriptive only (6 positive recordings from 5 participants, all in ALS).',fontsize=6.7,color=COL_MUTED)

fig.subplots_adjust(left=0.22,right=0.985,top=0.925,bottom=0.075)

stem='Figure_goal2_cohort_characterization_PUBLICATION'
for ext in ['png','pdf','svg']:
    kwargs={'dpi':600} if ext=='png' else {}
    fig.savefig(fig_dir / f'{stem}.{ext}', bbox_inches='tight', facecolor='white', **kwargs)
plt.close(fig)
print(fig_dir / f'{stem}.png')
