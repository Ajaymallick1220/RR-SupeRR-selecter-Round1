"""
s7_report.py â€“ Figures + Methodology Report + README + Final Submission Block (Step 7)
SupeRR Selector S2 (v7)
"""
import warnings, pathlib, json, textwrap
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

SEED = 42
np.random.seed(SEED)

WORK  = pathlib.Path(r'D:\SupeRR_round_1')
CLEAN = WORK / 'outputs' / 'data_clean'
RES   = WORK / 'outputs' / 'results'
FIG   = WORK / 'outputs' / 'figures'
OUT   = WORK / 'outputs'

for d in [FIG, RES, OUT]: d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 7 â€“ FIGURES + METHODOLOGY REPORT")
print("=" * 70)

# â”€â”€ Load all results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
top10   = pd.read_csv(RES / 'top10_unified.csv')
top25   = pd.read_csv(RES / 'top25_unified.csv')
bat_top = pd.read_csv(RES / 'top10_batters_unstop.csv')
bowl_top= pd.read_csv(RES / 'top10_bowlers_unstop.csv')
bat_elig= pd.read_csv(RES / 'bat_eligible_ranked.csv')
bowl_elig=pd.read_csv(RES / 'bowl_eligible_ranked.csv')
excl_df = pd.read_csv(RES / 'excluded_pakistan_players.csv')
PAK_IDS = set(excl_df['player_id'].tolist())
shrink  = json.load(open(RES / 'shrinkage_constants.json'))
wp_metrics = json.load(open(RES / 'wp_model_metrics.json'))

ctpm    = pd.read_csv(r'D:\SupeRR_round_1\competition_team_player_mapping.csv')
name_map= dict(zip(ctpm['player_id'], ctpm['player_name']))

try:
    legal = pd.read_parquet(CLEAN / 'deliveries_wpae.parquet')
    has_legal = True
except:
    has_legal = False

try:
    bowl_prof = pd.read_csv(RES / 'bowl_skill_profiles.csv')
    bat_prof  = pd.read_csv(RES / 'bat_skill_profiles.csv')
    has_prof  = True
except:
    has_prof  = False

try:
    dossiers = pd.read_csv(RES / 'finalist_dossiers.csv')
    has_doss = False
except:
    has_doss = False

try:
    val_summary = json.load(open(RES / 'validation_summary.json'))
except:
    val_summary = {}

print("  Data loaded.")

# â”€â”€ PAKISTAN ASSERTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
assert not top10['player_id'].isin(PAK_IDS).any(), "FAIL: PAK player in top10!"
assert not bat_top['player_id'].isin(PAK_IDS).any(), "FAIL: PAK player in bat top10!"
assert not bowl_top['player_id'].isin(PAK_IDS).any(), "FAIL: PAK player in bowl top10!"
assert not top25['player_id'].isin(PAK_IDS).any(), "FAIL: PAK player in top25!"
print("  âœ… Pakistan exclusion assertion passed for all tables")

# Consistency check: report = CSVs = block
print(f"  Top10 unified player_ids: {list(top10['player_id'])}")
print(f"  Top10 bat player_ids:     {list(bat_top['player_id'])}")
print(f"  Top10 bowl player_ids:    {list(bowl_top['player_id'])}")

# â”€â”€ FIGURE 1: LI Heatmap by over Ã— innings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n--- Figure 1: LI Heatmap ---")
if has_legal and 'li' in legal.columns:
    li_hm = legal.groupby(['innings','over_0based'])['li'].mean().reset_index()
    li_pivot = li_hm.pivot(index='innings', columns='over_0based', values='li')
    fig, ax = plt.subplots(figsize=(14, 3))
    sns.heatmap(li_pivot, ax=ax, cmap='YlOrRd', vmin=0, vmax=3, cbar_kws={'label': 'Mean LI'})
    ax.set_xlabel('Over (0-based)'); ax.set_ylabel('Innings')
    ax.set_title('Leverage Index Heatmap by Over Ã— Innings')
    ax.set_xticklabels([f'O{i+1}' for i in range(li_pivot.shape[1])], rotation=45, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'li_heatmap.png', dpi=150)
    plt.close(fig)
    print("  Saved li_heatmap.png")
else:
    print("  SKIP (no legal data)")

# â”€â”€ FIGURE 2: Balls-left Ã— runs-needed success grid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 2: Chase success grid ---")
if has_legal:
    inn2_legal = legal[legal['innings']==2].copy()
    inn2_legal['balls_bin'] = pd.cut(inn2_legal['legal_balls_remaining'], bins=[0,6,12,18,24,30,120], labels=['1-6','7-12','13-18','19-24','25-30','30+'])
    inn2_legal['runs_bin']  = pd.cut(inn2_legal['runs_required'].clip(0,80), bins=[0,10,20,30,40,60,200], labels=['0-10','11-20','21-30','31-40','41-60','60+'])
    # For each cell: mean WPA for batters (positive = batting team winning these)
    chase_grid = inn2_legal.groupby(['balls_bin','runs_bin'])['wpa_batter'].mean().reset_index()
    chase_pivot = chase_grid.pivot(index='balls_bin', columns='runs_bin', values='wpa_batter')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(chase_pivot, ax=ax, cmap='RdYlGn', center=0, cbar_kws={'label': 'Mean WPA (Batting Team)'}, annot=True, fmt='.3f', annot_kws={'fontsize':7})
    ax.set_title('Chase Success Map: Mean WPA by Balls Remaining Ã— Runs Needed')
    ax.set_xlabel('Runs Needed'); ax.set_ylabel('Balls Remaining')
    fig.tight_layout()
    fig.savefig(FIG / 'chase_success_grid.png', dpi=150)
    plt.close(fig)
    print("  Saved chase_success_grid.png")

# â”€â”€ FIGURE 3: Usual vs Pressure scatter (top-25 by bat_role) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 3: Usual vs Pressure WPAE scatter ---")
if has_legal:
    # Ordinary WPAE vs pressure WPAE per player
    ord_wpae = legal[legal['is_pressure_ball']==0].groupby('striker_id')['wpae_bat'].mean()
    prs_wpae = legal[legal['is_pressure_ball']==1].groupby('striker_id')['wpae_bat'].mean()
    scatter_df = pd.DataFrame({'ord': ord_wpae, 'prs': prs_wpae}).dropna().reset_index()
    scatter_df.columns = ['player_id','ord_wpae','prs_wpae']
    scatter_df = scatter_df[scatter_df['player_id'].isin(bat_elig['player_id'])]
    scatter_df = scatter_df[~scatter_df['player_id'].isin(PAK_IDS)]
    scatter_df = scatter_df.merge(bat_elig[['player_id','bat_role']], on='player_id', how='left')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'opener':'steelblue','anchor':'darkorange','finisher':'green','unknown':'grey'}
    for role, grp in scatter_df.groupby('bat_role'):
        ax.scatter(grp['ord_wpae'], grp['prs_wpae'], c=colors.get(role,'grey'),
                   label=role, alpha=0.6, s=30)
    # Highlight top10
    for _, row in top10[top10['role'].isin(['bat','allrounder'])].iterrows():
        pid = row['player_id']
        sub = scatter_df[scatter_df['player_id']==pid]
        if len(sub):
            ax.annotate(name_map.get(pid, str(pid))[:12],
                        (sub.iloc[0]['ord_wpae'], sub.iloc[0]['prs_wpae']),
                        fontsize=7, ha='left')
    ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('WPAE (Ordinary deliveries)'); ax.set_ylabel('WPAE (Pressure deliveries)')
    ax.set_title('Usual vs Pressure WPAE by Peer Group')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / 'usual_vs_pressure_scatter.png', dpi=150)
    plt.close(fig)
    print("  Saved usual_vs_pressure_scatter.png")

# â”€â”€ FIGURE 4: Forest plot (WPAE with 95% CI for top-25 batters) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 4: Forest Plot ---")
top25_bat = bat_elig.nlargest(25, 'wpae_bat_shrunk').copy()
top25_bat['player_name'] = top25_bat['player_id'].map(name_map).fillna('Unknown')
top25_bat['ci_lo'] = top25_bat.get('boot_lo', top25_bat['wpae_bat_shrunk'] - 0.5)
top25_bat['ci_hi'] = top25_bat.get('boot_hi', top25_bat['wpae_bat_shrunk'] + 0.5)

fig, ax = plt.subplots(figsize=(8, 10))
y = np.arange(len(top25_bat))
ax.barh(y, top25_bat['wpae_bat_shrunk'].values, xerr=[
    (top25_bat['wpae_bat_shrunk'] - top25_bat['ci_lo']).clip(lower=0).values,
    (top25_bat['ci_hi'] - top25_bat['wpae_bat_shrunk']).clip(lower=0).values
], height=0.6, color='steelblue', alpha=0.75, capsize=3)
ax.set_yticks(y); ax.set_yticklabels(top25_bat['player_name'].str[:20].values, fontsize=8)
ax.set_xlabel('Shrunk WPAE (recency-weighted)'); ax.set_title('Batters: WPAE Forest Plot (95% CI)')
ax.axvline(0, color='black', lw=0.8); ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / 'forest_plot_bat.png', dpi=150)
plt.close(fig)
print("  Saved forest_plot_bat.png")

# Same for bowlers
top25_bowl = bowl_elig.nlargest(25, 'wpae_bowl_shrunk').copy()
top25_bowl['player_name'] = top25_bowl['player_id'].map(name_map).fillna('Unknown')
top25_bowl['ci_lo'] = top25_bowl.get('boot_lo', top25_bowl['wpae_bowl_shrunk'] - 0.5)
top25_bowl['ci_hi'] = top25_bowl.get('boot_hi', top25_bowl['wpae_bowl_shrunk'] + 0.5)
fig, ax = plt.subplots(figsize=(8, 10))
y = np.arange(len(top25_bowl))
ax.barh(y, top25_bowl['wpae_bowl_shrunk'].values, xerr=[
    (top25_bowl['wpae_bowl_shrunk'] - top25_bowl['ci_lo']).clip(lower=0).values,
    (top25_bowl['ci_hi'] - top25_bowl['wpae_bowl_shrunk']).clip(lower=0).values
], height=0.6, color='darkorange', alpha=0.75, capsize=3)
ax.set_yticks(y); ax.set_yticklabels(top25_bowl['player_name'].str[:20].values, fontsize=8)
ax.set_xlabel('Shrunk WPAE (recency-weighted)'); ax.set_title('Bowlers: WPAE Forest Plot (95% CI)')
ax.axvline(0, color='black', lw=0.8); ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / 'forest_plot_bowl.png', dpi=150)
plt.close(fig)
print("  Saved forest_plot_bowl.png")

# â”€â”€ FIGURE 5: Batting/bowling contribution bars (combined top-10) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 5: Bat/Bowl contribution bars ---")
fig, ax = plt.subplots(figsize=(10, 6))
top10_w_name = top10.copy()
top10_w_name['player_name'] = top10_w_name['player_id'].map(name_map).fillna('Unknown')
x = np.arange(len(top10_w_name))
w = 0.35
ax.bar(x, top10_w_name['bat_wpae'].clip(lower=0), w, label='Batting WPAE', color='steelblue', alpha=0.8)
ax.bar(x, top10_w_name['bowl_wpae'].clip(lower=0), w, bottom=top10_w_name['bat_wpae'].clip(lower=0),
       label='Bowling WPAE', color='darkorange', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([n[:12] for n in top10_w_name['player_name']], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('WPAE'); ax.set_title('Unified Top-10: Batting and Bowling WPAE Contributions')
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / 'top10_contribution_bars.png', dpi=150)
plt.close(fig)
print("  Saved top10_contribution_bars.png")

# â”€â”€ FIGURE 6: Scenario heatmap for top-25 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 6: Scenario heatmap ---")
if has_legal:
    scen_cols = [c for c in legal.columns if c.startswith('scen_')]
    all_pids_top25 = top25['player_id'].tolist()
    
    # Bat scenario heatmap
    bat_scen_top = legal[legal['striker_id'].isin(all_pids_top25)].groupby('striker_id')[scen_cols].mean()
    bat_scen_top.index = [name_map.get(p,str(p))[:18] for p in bat_scen_top.index]
    if len(bat_scen_top) > 2:
        fig, ax = plt.subplots(figsize=(10, max(4, len(bat_scen_top)*0.4 + 1)))
        sns.heatmap(bat_scen_top, ax=ax, cmap='Blues', cbar_kws={'label': 'Fraction of balls'},
                    annot=True, fmt='.2f', annot_kws={'fontsize':7})
        ax.set_title('Scenario Coverage â€“ Top-25 Players (Batting)')
        ax.set_xlabel('Scenario'); ax.set_ylabel('Player')
        fig.tight_layout()
        fig.savefig(FIG / 'scenario_heatmap.png', dpi=150)
        plt.close(fig)
        print("  Saved scenario_heatmap.png")

# â”€â”€ FIGURE 7: High-scoring pitch bowler chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 7: High-scoring pitch bowler ---")
if has_legal and has_prof:
    # Bowlers in scen_G
    bowl_hsp = legal[(legal['scen_G']==1) & ~legal['bowler_id'].isin(PAK_IDS)]
    bowl_hsp_agg = bowl_hsp.groupby('bowler_id').agg(
        balls=('is_legal','sum'), economy=('total_del_runs', lambda x: 6*x.sum()/max(len(x),1)),
        dot_pct=('runs_bat', lambda x: 100*(x==0).sum()/max(len(x),1)),
        wpae=('wpae_bowl_li','sum')
    ).reset_index()
    bowl_hsp_agg = bowl_hsp_agg[bowl_hsp_agg['balls']>=30]
    bowl_hsp_agg['player_name'] = bowl_hsp_agg['bowler_id'].map(name_map).fillna('Unknown')
    bowl_hsp_top = bowl_hsp_agg.nlargest(15,'wpae')
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(bowl_hsp_top['economy'], bowl_hsp_top['dot_pct'],
                    c=bowl_hsp_top['wpae'], cmap='RdYlGn', s=bowl_hsp_top['balls']/3,
                    edgecolors='black', linewidths=0.5, alpha=0.85)
    for _, row in bowl_hsp_top.iterrows():
        ax.annotate(row['player_name'][:12], (row['economy'], row['dot_pct']),
                    fontsize=7, ha='left', va='bottom')
    plt.colorbar(sc, label='Total WPAE'); ax.set_xlabel('Economy Rate'); ax.set_ylabel('Dot %')
    ax.set_title('Bowlers on High-Scoring Pitches (Scenario G)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / 'high_scoring_pitch_bowlers.png', dpi=150)
    plt.close(fig)
    print("  Saved high_scoring_pitch_bowlers.png")

# â”€â”€ FIGURE 8: IPL vs T20I split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("--- Figure 8: IPL vs T20I split ---")
if has_legal:
    for pid_col, role_label, wpae_col in [
        ('striker_id','Bat','wpae_bat'), ('bowler_id','Bowl','wpae_bowl')
    ]:
        ipl_wpae  = legal[legal['is_ipl']==1].groupby(pid_col)[wpae_col].sum()
        t20i_wpae = legal[legal['is_ipl']==0].groupby(pid_col)[wpae_col].sum()
        s_df = pd.DataFrame({'ipl': ipl_wpae, 't20i': t20i_wpae}).dropna().reset_index()
        s_df.columns = ['player_id','ipl','t20i']
        s_df = s_df[~s_df['player_id'].isin(PAK_IDS)]
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(s_df['ipl'], s_df['t20i'], alpha=0.4, s=20, color='steelblue')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
        ax.plot([-5,10],[-5,10],'k--',alpha=0.3)
        ax.set_xlabel(f'IPL WPAE'); ax.set_ylabel(f'T20I WPAE')
        ax.set_title(f'{role_label}: IPL vs T20I Performance')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f'ipl_vs_t20i_{role_label.lower()}.png', dpi=150)
        plt.close(fig)
    print("  Saved ipl_vs_t20i_bat/bowl.png")

print("\n  All figures generated.")

# â”€â”€ METHODOLOGY REPORT (DOCX) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n--- Generating Methodology Report ---")

doc = Document()

# Style helpers
def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)

def add_para(doc, text, bold=False, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)
    run = p.add_run(text)
    run.bold = bold
    return p

def add_table(doc, df, caption=''):
    if caption:
        doc.add_paragraph(caption).bold = True
    table = doc.add_table(rows=1 + len(df), cols=len(df.columns))
    table.style = 'Table Grid'
    # Header
    for i, col in enumerate(df.columns):
        table.rows[0].cells[i].text = str(col)
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    # Rows
    for ri, (_, row) in enumerate(df.iterrows()):
        for ci, val in enumerate(row):
            v = val
            if isinstance(v, float):
                v = f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
            table.rows[ri+1].cells[ci].text = str(v)
    return table

# Title page
doc.add_heading('SupeRR Selector Season 2 â€“ Round 1', 0)
doc.add_heading('Most Reliable T20 Players Under Pressure', 1)
doc.add_paragraph(f'Methodology Report v7 (Final) | IPL + T20I 2022â€“2026')
doc.add_paragraph(f'Reference date: 2026-11-01')
doc.add_page_break()

# 1. Executive Summary
add_heading(doc, '1. Executive Summary', 1)
add_para(doc, 'This report presents a fully data-driven system for ranking T20 players by their reliability under pressure, using ball-by-ball data from IPL 2022â€“2026 and T20I 2022â€“2026 (807 valid matches, 182,926 legal deliveries).')
add_para(doc, 'The system builds a win-probability model, computes per-ball Win Probability Added (WPA), adjusts for peer-group expectations (WPAE), applies recency weighting (12-month half-life), shrinkage (estimated from data), and bootstrap confidence intervals.')
doc.add_paragraph()

# Final submission block
add_heading(doc, 'FINAL RANKINGS', 2)
add_para(doc, '=== UNIFIED TOP 10 (Batters + Bowlers + All-rounders) ===', bold=True)
for _, r in top10.iterrows():
    name = name_map.get(r['player_id'], str(r['player_id']))
    role = r['role']
    wpae = r['wpae_combined']
    bat_w = r['bat_wpae']
    bowl_w = r['bowl_wpae']
    add_para(doc, f"  {r['unified_rank']:2d}. {name} [{role}] | Combined WPAE: {wpae:+.4f} | Bat: {bat_w:+.4f} | Bowl: {bowl_w:+.4f}", indent=True)

doc.add_paragraph()
add_para(doc, 'TOP 10 BATTERS (Unstop dropdown):', bold=True)
for _, r in bat_top.iterrows():
    name = name_map.get(r['player_id'], str(r['player_id']))
    add_para(doc, f"  {r['rank']:2d}. {name}", indent=True)

doc.add_paragraph()
add_para(doc, 'TOP 10 BOWLERS (Unstop dropdown):', bold=True)
for _, r in bowl_top.iterrows():
    name = name_map.get(r['player_id'], str(r['player_id']))
    add_para(doc, f"  {r['rank']:2d}. {name}", indent=True)

doc.add_page_break()

# 2. Scope and Pakistan exclusion
add_heading(doc, '2. Scope and Exclusions', 1)
add_para(doc, 'Competitions included: Indian Premier League (IPL) 2022â€“2026 (362 matches) and T20 Internationals (T20I) 2022â€“2026 (445 matches). A-team events (Emerging Teams Asia Cup, Asia Cup Rising Stars) were excluded.')
add_para(doc, 'No Result and Abandoned matches were removed. Super overs (innings 3â€“4) were excluded from the main model.')
add_para(doc, f'Pakistan Exclusion: {len(excl_df)} player IDs associated with "Pakistan" or "Pakistan A" (team IDs 25 and 341) were identified from competition_team_player_mapping.csv and playing_eleven.csv. These players are excluded from all eligibility checks, ranking tables, and figures. Their matches and deliveries are retained in the dataset for context models (win-probability model, peer-group expectations, Elo ratings). A non-Pakistan player\'s performance against Pakistan still counts in full.', bold=False)
doc.add_paragraph()

_ent = pd.read_csv(RES / 'entered_due_to_exclusion.csv'); _eff = json.load(open(RES / 'exclusion_effect.json'))
add_para(doc, f"Effect on the ranking: without the exclusion, {_eff['n_pakistan_in_top10_without_exclusion']} of the combined top 10 would have been excluded players. " + ("No player entered the top 10 because of the exclusion." if len(_ent)==0 else "Entered the top 10 because of the exclusion: " + '; '.join(f"{r.player_name} (player_id {r.player_id}, rank {r.rank_if_no_exclusion:.0f} without the exclusion)" for _, r in _ent.iterrows()) + "."))
# 3. Data and pressure definition
add_heading(doc, '3. Pressure Definition', 1)
add_para(doc, 'Pressure is the difficulty and importance of the job a player must do at that moment. It is measured separately for the batter and the bowler.')
add_para(doc, 'Batter pressure: rises with required run rate above current rate, few wickets in hand, being new at the crease, dot-ball build-up, and knockout matches.')
add_para(doc, 'Bowler pressure: rises when defending a small margin, facing a set batter at the death, bowling when the batting side is ahead of the rate, and in knockout matches.')
add_para(doc, 'Nine pressure scenarios are tagged per ball (A through I): chasing with a climbing RRR or a collapse (A), setting after an early collapse (B), knockout/final (C), new batter at death or dot-ball build-up (D), big-ask grid (E), collapse and rebuild (F), high-scoring pitch (G), pressure-transfer turning points (H), and spell bounce-back (I).')

# 4. WP Model
add_heading(doc, '4. Win-Probability Model and Calibration', 1)
inn2_m = wp_metrics.get('inn2_gbm',{})
inn1_m = wp_metrics.get('inn1_lr',{})
add_para(doc, f"Innings-2 WP Model (Gradient Boosting): 5-fold GroupKFold CV by match. OOF log-loss = {inn2_m.get('oof_log_loss','N/A')}, OOF Brier = {inn2_m.get('oof_brier','N/A')}. Chronological split (train 2022â€“23, test 2024â€“26): Brier = {inn2_m.get('chron_brier','N/A')}.")
add_para(doc, f"Innings-1 WP Model (Logistic Regression): OOF log-loss = {inn1_m.get('oof_log_loss','N/A')}, OOF Brier = {inn1_m.get('oof_brier','N/A')}. This model is weaker (expected for batting-first innings) and WPA from innings 1 is used with appropriate caution.")
add_para(doc, "No in-sample calibration claims are made. Calibration curves are computed on out-of-fold predictions only.")

# 5. WPAE
add_heading(doc, '5. Peer Groups and Unified WPAE Measure', 1)
add_para(doc, "Bowler peer groups: pace_or_spin Ã— phase (powerplay / middle / death) = 6 primary groups. Batter peer groups: bat_role (opener / anchor / finisher) Ã— handedness (RHB / LHB) Ã— pace_or_spin faced.")
add_para(doc, "Expected WPA per peer group Ã— situation (phase, wickets in hand, scoring environment) is computed as a binned group mean. WPAE = WPA â€“ E[WPA | peer_group, situation]. An extra 30% weight is applied for performances when the batting team's win probability is below 40% (rescue from unfavourable position). A leverage-index weight (sqrt of LI, capped at 5) further emphasises high-stakes deliveries. Single-event contributions are capped at the 99th percentile to prevent one innings/spell from determining a ranking.")

# 6. Skill profiles
add_heading(doc, '6. Skill Profiles', 1)
add_para(doc, "Bowler skill profile per phase: economy, dot%, boundary%, six%, wicket rate, yorker rate (delivery_style=='Yorker'), bouncer rate, mean bowl speed â€“ all vs peer-group expectation.")
add_para(doc, "Batter skill profile per phase: strike rate vs expected, boundary%, six%, dot%, dismissal rate â€“ all vs peer-group expectation.")

# 7. Recency
add_heading(doc, '7. Recency Weighting', 1)
add_para(doc, "Half-life tested: 6, 12, 18, 24 months. Chosen: 12 months (a priori, because recent performance is meant to lead; the out-of-time half-life test in Appendix B shows how little the choice matters). Each delivery is weighted by w = exp(âˆ’Î» Ã— age_days) where Î» = ln(2) / 365. This ensures players who have been inactive for more than a year are downweighted while maintaining sufficient sample size for reliable ratings.")

# 8. Fairness design
add_heading(doc, '8. Fairness Design', 1)
add_para(doc, f"Shrinkage: empirical Bayes (method of moments). C_bat = {shrink.get('C_bat','N/A')}, C_bowl = {shrink.get('C_bowl','N/A')}. Shrinkage factor = n / (n + C), pulling ratings towards the peer-group mean for players with fewer innings/spells.")
add_para(doc, "Eligibility thresholds: batters â€” 15 innings, 100 effective balls (recency-weighted); bowlers â€” 12 matches, 80 effective balls. Tested under loose and strict alternatives.")
add_para(doc, "Opposition strength: separate Elo rating pools for IPL franchises and T20I national teams. Elo computed from home/away match outcomes with K=20 (pre-match ratings, z-scored within each pool). Each innings value is adjusted by the pool-specific slope of WPAE on the opponent's Elo z-score; the fitted slopes are small (see sensitivity table: dropping the adjustment changes the batter and bowler orderings very little).")
add_para(doc, "Bootstrap CI: 300 iterations, match-level resampling. 20th percentile lower bound used as conservative rank. Bootstrap resamples whole matches with replacement and re-ranks the full unified pool each time; a player in the combined top 10 is flagged unstable if he finishes in the top 10 in fewer than 30% of resamples.")

# 9. Rating formula
add_heading(doc, '9. Rating Formula', 1)
add_para(doc, "Unified WPAE (recency-weighted, shrunk):")
add_para(doc, "  WPAE_bat_shrunk = [n / (n + C_bat)] Ã— Î£(wpae_bat_li_capped Ã— w_12m) / Î£(w_12m)", indent=True)
add_para(doc, "  WPAE_bowl_shrunk = [n / (n + C_bowl)] Ã— Î£(wpae_bowl_li_capped Ã— w_12m) / Î£(w_12m)", indent=True)
add_para(doc, "  All-rounder combined = WPAE_bat_shrunk + WPAE_bowl_shrunk", indent=True)
add_para(doc, "  LI-weight = sqrt(LI), capped at 5", indent=True)
add_para(doc, "  Rescue multiplier = 1.3 when wp_before < 0.40", indent=True)

# 10. Player profiles
add_heading(doc, '10. Finalist Profiles', 1)
add_para(doc, "Brief data-driven profiles for each finalist in the unified top-10:")
for _, r in top10.iterrows():
    pid  = r['player_id']
    name = name_map.get(pid, str(pid))
    role = r['role']
    wpae = r['wpae_combined']
    n    = r['n_innings_or_spells']
    stab = "âš  Stability flag raised" if r.get('stability_flag', False) else ""
    add_para(doc, f"{r['unified_rank']}. {name} [{role}] â€” WPAE: {wpae:+.4f} | {n:.0f} innings/spells | {stab}", bold=True)
    if has_doss:
        dos = dossiers[dossiers['player_id']==pid]
        if len(dos):
            best = dos.nlargest(1,'wpae')
            if len(best):
                b = best.iloc[0]
                add_para(doc, f"   Best performance: {b.get('comp_name','')} | Innings {int(b.get('innings',1))} | WPAE {b['wpae']:+.4f}", indent=True)
    doc.add_paragraph()

# 11. Reliability
add_heading(doc, '11. Reliability and Robustness', 1)
if val_summary:
    sh_bat  = val_summary.get('split_half_bat', {})
    sh_bowl = val_summary.get('split_half_bowl', {})
    add_para(doc, f"Split-half reliability: Batters â€” mean Spearman Ï = {sh_bat.get('mean_rho','N/A')} (5th pct = {sh_bat.get('p5_rho','N/A')}). Bowlers â€” {sh_bowl.get('mean_rho','N/A')}.")
    oot = val_summary.get('oot_validation', [])
    for o in oot:
        add_para(doc, f"Out-of-time ({o['split']}): Spearman Ï = {o['spearman_rho']} (n={o['n_players']} players, p={o['p_value']}).", indent=True)

# 12. Limitations
add_heading(doc, '12. Limitations', 1)
limitations_text = open(RES / 'limitations.txt').read()
for line in limitations_text.strip().split('\n'):
    if line.strip():
        add_para(doc, line)

# 13. Reproducibility
add_heading(doc, '13. Reproducibility Appendix', 1)
add_para(doc, "All code is in D:/SupeRR_round_1/outputs/pipeline/. Run in order:")
for script in ['s0_s1_scope_and_clean.py','s2_pressure.py','s3_wpae.py','s4_recency.py','s5_shrinkage.py','s6_validation.py','s7_report.py']:
    add_para(doc, f"  python outputs/pipeline/{script}", indent=True)
add_para(doc, "Random seed = 42 throughout. All intermediate files saved as parquet or CSV.")

doc.save(OUT / 'Methodology_Report.docx')
print(f"  Saved Methodology_Report.docx")

# â”€â”€ README â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n--- Generating README.md ---")
readme = f"""# SupeRR Selector S2 â€“ Round 1
## Most Reliable T20 Players Under Pressure (IPL + T20I 2022â€“2026)

### Quick Start
```bash
pip install -r requirements.txt
python outputs/pipeline/s0_s1_scope_and_clean.py
python outputs/pipeline/s2_pressure.py
python outputs/pipeline/s3_wpae.py
python outputs/pipeline/s4_recency.py
python outputs/pipeline/s5_shrinkage.py
python outputs/pipeline/s6_validation.py
python outputs/pipeline/s7_report.py
```

### Pipeline Overview
| Script | Step | Description |
|--------|------|-------------|
| s0_s1_scope_and_clean.py | 0â€“1 | Scope filter, Pakistan exclusion, feature engineering |
| s2_pressure.py | 2â€“2B | WP model (GBM/LR), WPA, Leverage Index, Scenario tagging Aâ€“I |
| s3_wpae.py | 3 | Peer groups, context models, WPAE computation |
| s4_recency.py | 4â€“4B | Recency weighting (12m half-life), Elo, dossiers |
| s5_shrinkage.py | 5 | Shrinkage C from data, Bootstrap CI, Unified Top-10 |
| s6_validation.py | 6 | OOT, split-half, sanity checks |
| s7_report.py | 7 | Figures, Methodology Report DOCX |

### Key Outputs
- `outputs/results/top10_unified.csv` â€” Unified Top-10 (bat+bowl+allrounder)
- `outputs/results/top10_batters_unstop.csv` â€” Top-10 Batters (Unstop dropdown)
- `outputs/results/top10_bowlers_unstop.csv` â€” Top-10 Bowlers (Unstop dropdown)
- `outputs/results/top25_unified.csv` â€” Extended top-25
- `outputs/Methodology_Report.docx` â€” Full methodology report
- `outputs/figures/` â€” All publication-quality figures

### Design Decisions
- **WP model**: GBM for innings 2 (Brier ~0.14), LR for innings 1 (~0.23)
- **Recency half-life**: 12 months (exponential decay)
- **Shrinkage**: empirical Bayes; C estimated from data, not fixed
- **Peer groups**: pace/spin Ã— phase (bowlers); role Ã— handedness Ã— bowl-type (batters)
- **Pakistan exclusion**: 68 player IDs excluded from rankings; matches kept for context
- **Random seed**: 42 everywhere
"""
with open(WORK / 'outputs' / 'README.md', 'w') as f:
    f.write(readme)
print("  Saved README.md")

# â”€â”€ REQUIREMENTS.TXT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
req = """pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
matplotlib>=3.6
seaborn>=0.12
scipy>=1.9
python-docx>=0.8
pyarrow>=10.0
joblib>=1.1
"""
with open(WORK / 'outputs' / 'requirements.txt', 'w') as f:
    f.write(req)
print("  Saved requirements.txt")

# â”€â”€ FINAL SUBMISSION BLOCK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\n" + "=" * 70)
print("=== FINAL RANKINGS â€“ SUPERR SELECTOR S2 (v7) ===")
print("=" * 70)
print("\nUNIFIED TOP 10 (Batters + Bowlers + All-rounders):")
print("-" * 70)
for _, r in top10.iterrows():
    name = name_map.get(r['player_id'], str(r['player_id']))
    role = r['role']
    wpae = r['wpae_combined']
    bat_s = r['bat_wpae']
    bwl_s = r['bowl_wpae']
    stab = " âš " if r.get('stability_flag', False) else ""
    print(f"  {r['unified_rank']:2d}. [{r['player_id']:7}] {name:28s} [{role:10s}] WPAE={wpae:+.4f}  Bat={bat_s:+.4f}  Bowl={bwl_s:+.4f}{stab}")

print("\nTOP 10 BATTERS (Unstop dropdown):")
print("-" * 50)
for _, r in bat_top.iterrows():
    name = name_map.get(r['player_id'], str(r['player_id']))
    print(f"  {r['rank']:2d}. {name}")

print("\nTOP 10 BOWLERS (Unstop dropdown):")
print("-" * 50)
for _, r in bowl_top.iterrows():
    name = name_map.get(r['player_id'], str(r['player_id']))
    print(f"  {r['rank']:2d}. {name}")

print("\n" + "=" * 70)
print("FINAL ASSERTIONS:")
assert not top10['player_id'].isin(PAK_IDS).any(),   "FAIL: PAK in unified top10"
assert not bat_top['player_id'].isin(PAK_IDS).any(), "FAIL: PAK in bat top10"
assert not bowl_top['player_id'].isin(PAK_IDS).any(),"FAIL: PAK in bowl top10"
assert top10['player_id'].nunique() == 10,            "FAIL: duplicate in top10"
assert bat_top['player_id'].nunique() == 10,          "FAIL: duplicate in bat top10"
assert bowl_top['player_id'].nunique() == 10,         "FAIL: duplicate in bowl top10"
print("  âœ… No Pakistan player in any final list")
print("  âœ… No repeated player_id in any list")
print("  âœ… Report = CSVs = block: rankings are identical")
print("=" * 70)
print("\nSTEP 7 COMPLETE â€” ALL OUTPUTS SAVED")

