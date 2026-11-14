"""
s4_recency.py – Recency Weighting + Dossiers (Steps 4 & 4B)
SupeRR Selector S2 (v7)
"""
import warnings, pathlib, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SEED = 42
np.random.seed(SEED)

WORK  = pathlib.Path(r'D:\SupeRR_round_1')
CLEAN = WORK / 'outputs' / 'data_clean'
RES   = WORK / 'outputs' / 'results'
FIG   = WORK / 'outputs' / 'figures'

print("=" * 70)
print("STEP 4 – RECENCY WEIGHTING + ELIGIBILITY")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading data …")
bat_mi   = pd.read_parquet(CLEAN / 'bat_match_innings.parquet')
bowl_mi  = pd.read_parquet(CLEAN / 'bowl_match_innings.parquet')
legal    = pd.read_parquet(CLEAN / 'deliveries_wpae.parquet')
excl_df  = pd.read_csv(RES / 'excluded_pakistan_players.csv')
PAK_IDS  = set(excl_df['player_id'].tolist())

# Load name map from competition_team_player_mapping
ctpm = pd.read_csv(r'D:\SupeRR_round_1\competition_team_player_mapping.csv')
name_map = dict(zip(ctpm['player_id'], ctpm['player_name']))

REF_DATE = pd.Timestamp('2026-11-01')
bat_mi['match_date']  = pd.to_datetime(bat_mi['match_date'])
bowl_mi['match_date'] = pd.to_datetime(bowl_mi['match_date'])

print(f"  Bat MI: {len(bat_mi):,}   Bowl MI: {len(bowl_mi):,}")
print(f"  Reference date: {REF_DATE.date()}")

# ── RECENCY WEIGHTING – test half-lives ───────────────────────────────────────
print("\n--- Testing Half-Lives (6, 12, 18, 24 months) ---")

HALF_LIVES_MONTHS = [6, 12, 18, 24]
LN2 = np.log(2)

def add_recency_weight(mi_df, date_col='match_date', half_life_months=12):
    hl_days = half_life_months * 30.44
    lam = LN2 / hl_days
    age_days = (REF_DATE - mi_df[date_col]).dt.days.clip(lower=0).astype(float)
    return np.exp(-lam * age_days)

# For each half-life: compute weighted sum for batters and bowlers
# Evaluate recency by split: compare 2022-23 period ranking vs 2024+ ranking (Spearman correlation)
early_cut = pd.Timestamp('2024-01-01')

results_hl = []
for hl in HALF_LIVES_MONTHS:
    # Batter
    bat_mi['w'] = add_recency_weight(bat_mi, half_life_months=hl)
    bat_full = bat_mi.groupby('striker_id').apply(
        lambda g: pd.Series({
            'weighted_wpae': (g['wpae_bat_li_sum_capped'] * g['w']).sum() / max(g['w'].sum(), 1e-9),
            'n_innings': len(g)
        })
    ).reset_index()
    bat_early = bat_mi[bat_mi['match_date'] < early_cut].groupby('striker_id')['wpae_bat_li_sum_capped'].mean().reset_index()
    bat_late  = bat_mi[bat_mi['match_date'] >= early_cut].groupby('striker_id')['wpae_bat_li_sum_capped'].mean().reset_index()
    merged_b  = bat_early.merge(bat_late, on='striker_id', suffixes=('_early','_late'))
    merged_b  = merged_b.merge(bat_full[['striker_id','weighted_wpae']], on='striker_id', how='left')
    rho_b, _  = spearmanr(merged_b['wpae_bat_li_sum_capped_early'], merged_b['weighted_wpae']) if len(merged_b)>10 else (0, 1)
    
    # Bowler
    bowl_mi['w'] = add_recency_weight(bowl_mi, half_life_months=hl)
    bowl_full = bowl_mi.groupby('bowler_id').apply(
        lambda g: pd.Series({
            'weighted_wpae': (g['wpae_bowl_li_sum_capped'] * g['w']).sum() / max(g['w'].sum(), 1e-9),
            'n_spells': len(g)
        })
    ).reset_index()
    results_hl.append({'half_life_months': hl, 'spearman_bat': round(rho_b, 3), 'n_bat': len(merged_b)})
    print(f"  HL={hl:2d}m: bat_spearman={rho_b:.3f}  n_bat={len(merged_b)}")

# Choose half-life: 12 months (matches prompt requirement and T20 dynamics)
CHOSEN_HL = 12
print(f"\n  Chosen half-life: {CHOSEN_HL} months (balances recency with sample size)")

# ── COMPUTE RECENCY-WEIGHTED RATINGS ─────────────────────────────────────────
print("\n--- Computing Recency-Weighted Ratings ---")

bat_mi['w12'] = add_recency_weight(bat_mi, half_life_months=12)

# Weighted WPAE per batter
def bat_weighted_agg(g):
    w = g['w12']
    wpae_w = (g['wpae_bat_li_sum_capped'] * w).sum()
    balls_w = (g['balls_faced'] * w).sum()
    n_inn   = len(g)
    n_match = g['match_id'].nunique() if 'match_id' in g.columns else n_inn
    return pd.Series({
        'wpae_bat_weighted':   wpae_w,
        'balls_bat_weighted':  balls_w,
        'weight_sum':          w.sum(),
        'n_innings':           n_inn,
        'n_matches':           n_match,
        'last_match':          g['match_date'].max(),
        'bat_role':            g['bat_role'].mode().iloc[0] if len(g)>0 else 'anchor',
        'n_comps':             g['comp_name'].nunique() if 'comp_name' in g.columns else 1,
        # Last 12m subset
        'wpae_last12':         g.loc[g['match_date'] >= REF_DATE - pd.DateOffset(months=12), 'wpae_bat_li_sum_capped'].sum(),
        'inn_last12':          (g['match_date'] >= REF_DATE - pd.DateOffset(months=12)).sum(),
        # Last 24-36m
        'wpae_last24':         g.loc[g['match_date'] >= REF_DATE - pd.DateOffset(months=24), 'wpae_bat_li_sum_capped'].sum(),
        'inn_last24':          (g['match_date'] >= REF_DATE - pd.DateOffset(months=24)).sum(),
    })

bat_rated = bat_mi.groupby('striker_id').apply(bat_weighted_agg).reset_index()
bat_rated.rename(columns={'striker_id': 'player_id'}, inplace=True)
bat_rated['player_name'] = bat_rated['player_id'].map(name_map).fillna('Unknown')

bowl_mi['w12'] = add_recency_weight(bowl_mi, half_life_months=12)

def bowl_weighted_agg(g):
    w = g['w12']
    wpae_w = (g['wpae_bowl_li_sum_capped'] * w).sum()
    balls_w = (g['balls_bowled'] * w).sum()
    n_sp    = len(g)
    n_match = g['match_id'].nunique() if 'match_id' in g.columns else n_sp
    return pd.Series({
        'wpae_bowl_weighted':  wpae_w,
        'balls_bowl_weighted': balls_w,
        'weight_sum':          w.sum(),
        'n_spells':            n_sp,
        'n_matches':           n_match,
        'last_match':          g['match_date'].max(),
        'pace_or_spin':        g['pace_or_spin'].mode().iloc[0] if len(g)>0 else 'Pace',
        'bowling_type':        g['bowling_type'].mode().iloc[0] if len(g)>0 else 'Unknown',
        'n_comps':             g['comp_name'].nunique() if 'comp_name' in g.columns else 1,
        'wpae_last12':         g.loc[g['match_date'] >= REF_DATE - pd.DateOffset(months=12), 'wpae_bowl_li_sum_capped'].sum(),
        'spells_last12':       (g['match_date'] >= REF_DATE - pd.DateOffset(months=12)).sum(),
        'wpae_last24':         g.loc[g['match_date'] >= REF_DATE - pd.DateOffset(months=24), 'wpae_bowl_li_sum_capped'].sum(),
        'spells_last24':       (g['match_date'] >= REF_DATE - pd.DateOffset(months=24)).sum(),
    })

bowl_rated = bowl_mi.groupby('bowler_id').apply(bowl_weighted_agg).reset_index()
bowl_rated.rename(columns={'bowler_id': 'player_id'}, inplace=True)
bowl_rated['player_name'] = bowl_rated['player_id'].map(name_map).fillna('Unknown')

print(f"  Batters rated: {len(bat_rated)}")
print(f"  Bowlers rated: {len(bowl_rated)}")


# ── ASSOCIATE-ONLY EXCLUSION & PRESSURE BALL COUNTS (85+ balls under pressure) ──
ASSOC_TEAMS = {24, 26, 89, 272, 282, 283, 313, 315, 3886}  # NED, SCO, CAN, UAE, NEP, HKG, OMN, USA, JPN
bat_teams_map = legal.groupby('striker_id')['batting_team_id'].apply(lambda s: set(s.unique())).to_dict()
bowl_teams_map = legal.groupby('bowler_id')['bowling_team_id'].apply(lambda s: set(s.unique())).to_dict()
ASSOC_ONLY_BAT = {p for p, ts in bat_teams_map.items() if ts and ts.issubset(ASSOC_TEAMS)}
ASSOC_ONLY_BOWL = {p for p, ts in bowl_teams_map.items() if ts and ts.issubset(ASSOC_TEAMS)}
print(f"  Associate-only players excluded from selection: batters={len(ASSOC_ONLY_BAT)}, bowlers={len(ASSOC_ONLY_BOWL)}")

# Count pressure balls faced/bowled per player and max pressure balls in a single phase
if 'true_sit_pressure' not in legal.columns:
    legal['true_sit_pressure'] = (legal[['scen_A','scen_B','scen_C','scen_D','scen_E','scen_F','scen_H']].sum(axis=1) >= 1).astype(int)
bat_p_balls = legal.groupby('striker_id')['true_sit_pressure'].sum()
bat_p_phase_max = legal[legal['true_sit_pressure']==1].groupby(['striker_id','phase']).size().unstack(fill_value=0).max(axis=1)
bowl_p_balls = legal.groupby('bowler_id')['true_sit_pressure'].sum()
bat_rated['pressure_balls'] = bat_rated['player_id'].map(bat_p_balls).fillna(0).astype(int)
bat_rated['max_phase_pressure_balls'] = bat_rated['player_id'].map(bat_p_phase_max).fillna(0).astype(int)
bowl_rated['pressure_balls'] = bowl_rated['player_id'].map(bowl_p_balls).fillna(0).astype(int)

# ── ELIGIBILITY THRESHOLDS ────────────────────────────────────────────────────
print("\n--- Eligibility Thresholds ---")

# Derive from data: use 25th percentile of finalists' load
# Start with min effective weight (sum of weights) = 2 matches worth at current decay
# Equivalent to: weight_sum >= exp(-lam*0)*2 = 2.0 as absolute floor
# More practically: n_matches >= 10 and effective_innings >= 10

# Show ranking under 3 threshold values
thresholds = [
    {'label': 'Loose',  'bat_min_inn': 8,  'bowl_min_match': 8,  'bat_min_balls': 50,  'bowl_min_balls': 50},
    {'label': 'Medium', 'bat_min_inn': 15, 'bowl_min_match': 12, 'bat_min_balls': 100, 'bowl_min_balls': 80},
    {'label': 'Strict', 'bat_min_inn': 25, 'bowl_min_match': 20, 'bat_min_balls': 200, 'bowl_min_balls': 150},
]

for thr in thresholds:
    b_elig = bat_rated[
        (bat_rated['n_innings'] >= thr['bat_min_inn']) &
        (bat_rated['balls_bat_weighted'] >= thr['bat_min_balls']) &
        ~bat_rated['player_id'].isin(PAK_IDS | ASSOC_ONLY_BAT) & (bat_rated['pressure_balls'] >= 85) & (bat_rated['max_phase_pressure_balls'] >= 85)
    ]
    bw_elig = bowl_rated[
        (bowl_rated['n_spells'] >= thr['bowl_min_match']) &
        (bowl_rated['balls_bowl_weighted'] >= thr['bowl_min_balls']) &
        ~bowl_rated['player_id'].isin(PAK_IDS | ASSOC_ONLY_BOWL) & (bowl_rated['pressure_balls'] >= 80)
    ]
    print(f"  [{thr['label']}]  eligible batters={len(b_elig)}  eligible bowlers={len(bw_elig)}")

# Use Medium thresholds as primary
BAT_MIN_INN    = 15
BAT_MIN_BALLS  = 100
BOWL_MIN_MATCH = 12
BOWL_MIN_BALLS = 80

bat_elig = bat_rated[
    (bat_rated['n_innings'] >= BAT_MIN_INN) &
    (bat_rated['balls_bat_weighted'] >= BAT_MIN_BALLS) &
    ~bat_rated['player_id'].isin(PAK_IDS | ASSOC_ONLY_BAT) & (bat_rated['pressure_balls'] >= 85) & (bat_rated['max_phase_pressure_balls'] >= 85)
].copy()

bowl_elig = bowl_rated[
    (bowl_rated['n_spells'] >= BOWL_MIN_MATCH) &
    (bowl_rated['balls_bowl_weighted'] >= BOWL_MIN_BALLS) &
    ~bowl_rated['player_id'].isin(PAK_IDS | ASSOC_ONLY_BOWL) & (bowl_rated['pressure_balls'] >= 80)
].copy()

print(f"\n  FINAL eligible (Medium): batters={len(bat_elig)}  bowlers={len(bowl_elig)}")

# Flag weak recent form
bat_elig['recent_weak'] = (bat_elig['wpae_last12'] < 0) & (bat_elig['wpae_bat_weighted'] > 0)
bowl_elig['recent_weak'] = (bowl_elig['wpae_last12'] < 0) & (bowl_elig['wpae_bowl_weighted'] > 0)
print(f"  Flagged recent_weak: batters={bat_elig.recent_weak.sum()}  bowlers={bowl_elig.recent_weak.sum()}")

# ── CONSISTENCY SCORE ─────────────────────────────────────────────────────────
print("\n--- Consistency Scores ---")

# Consistency = share of pressure innings/spells with positive WPAE
def consistency_score(mi_df, pid_col, wpae_col, pressure_col):
    scores = {}
    for pid, g in mi_df.groupby(pid_col):
        pressure_inn = g[g[pressure_col] > 0.5]
        if len(pressure_inn) == 0:
            scores[pid] = 0.5
            continue
        scores[pid] = (pressure_inn[wpae_col] > 0).mean()
    return pd.Series(scores, name='consistency')

bat_consist  = consistency_score(bat_mi,  'striker_id',  'wpae_bat_li_sum_capped', 'is_pressure')
bowl_consist = consistency_score(bowl_mi, 'bowler_id', 'wpae_bowl_li_sum_capped', 'is_pressure')

bat_elig  = bat_elig.merge(bat_consist.reset_index().rename(columns={'index':'player_id','consistency':'consistency'}),
                            on='player_id', how='left')
bowl_elig = bowl_elig.merge(bowl_consist.reset_index().rename(columns={'index':'player_id','consistency':'consistency'}),
                             on='player_id', how='left')

# ── OPPOSITION STRENGTH (ELO) ─────────────────────────────────────────────────
print("\n--- Opposition Strength (Elo Ratings) ---")

# Separate Elo pools: IPL franchises and national T20I teams
matches = pd.read_csv(r'D:\SupeRR_round_1\matches.csv')
teams   = pd.read_csv(r'D:\SupeRR_round_1\teams.csv')
comps   = pd.read_csv(r'D:\SupeRR_round_1\competitions.csv')

matches = matches.merge(comps[['comp_id','match_format','comp_name']], on='comp_id', how='left')
matches['is_ipl_match'] = matches['comp_name'].str.contains('IPL', na=False)
matches['match_date'] = pd.to_datetime(matches['match_date'])
# Keep in-scope
matches = matches[~matches['result'].isin(['No Result', 'Match Abandoned', 'Abandoned'])]

def compute_elo(match_df, team_col_a, team_col_b, win_col, K=20, default=1500):
    elo = {}
    history = []
    for _, row in match_df.sort_values('match_date').iterrows():
        ta, tb = row[team_col_a], row[team_col_b]
        ea_elo = elo.get(ta, default)
        eb_elo = elo.get(tb, default)
        # Expected win prob
        exp_a = 1 / (1 + 10 ** ((eb_elo - ea_elo) / 400))
        # Actual
        win_a = 1.0 if row[win_col] == ta else 0.0
        # Update
        elo[ta] = ea_elo + K * (win_a       - exp_a)
        elo[tb] = eb_elo + K * ((1-win_a) - (1-exp_a))
        history.append({'match_id': row['match_id'], 'team_id': ta, 'elo': elo[ta]})
        history.append({'match_id': row['match_id'], 'team_id': tb, 'elo': elo[tb]})
    return elo, pd.DataFrame(history)

# IPL Elo
ipl_matches = matches[matches['is_ipl_match'] & matches['win_team_id'].notna()].copy()
ipl_elo, ipl_hist = compute_elo(ipl_matches, 'home_team_id', 'away_team_id', 'win_team_id')

# T20I Elo
t20i_matches = matches[matches['match_format']=='T20I'].copy()
t20i_matches = t20i_matches[t20i_matches['win_team_id'].notna()]
t20i_elo, t20i_hist = compute_elo(t20i_matches, 'home_team_id', 'away_team_id', 'win_team_id')

# Map bowling_team Elo as opposition strength
# Join legal deliveries with Elo at match time
def attach_opp_elo(legal_df, ipl_hist, t20i_hist):
    # Latest Elo per team per match (from history)
    all_hist = pd.concat([
        ipl_hist.assign(pool='ipl'),
        t20i_hist.assign(pool='t20i')
    ])
    # For each legal ball, bowling team = opposition for batter
    # batting team = opposition for bowler
    bowl_team_elo = all_hist.groupby(['match_id','team_id'])['elo'].last().reset_index()
    bowl_team_elo.columns = ['match_id','bowling_team_id','opp_elo']
    legal_df = legal_df.merge(bowl_team_elo, on=['match_id','bowling_team_id'], how='left')
    legal_df['opp_elo'] = legal_df['opp_elo'].fillna(1500)
    return legal_df

legal = attach_opp_elo(legal, ipl_hist, t20i_hist)
legal['opp_strength'] = (legal['opp_elo'] - 1500) / 100  # z-score-like

print(f"  IPL Elo teams: {len(ipl_elo)}")
print(f"  T20I Elo teams: {len(t20i_elo)}")
print(f"  Top 5 IPL Elo: {sorted(ipl_elo.items(), key=lambda x: -x[1])[:5]}")
print(f"  Top 5 T20I Elo: {sorted(t20i_elo.items(), key=lambda x: -x[1])[:5]}")

# ── SAVE ELIGIBLE PLAYER LISTS ────────────────────────────────────────────────
bat_elig.to_csv(RES / 'bat_eligible.csv', index=False)
bowl_elig.to_csv(RES / 'bowl_eligible.csv', index=False)
legal.to_parquet(CLEAN / 'deliveries_opp_elo.parquet', index=False)

print(f"\n  Saved bat_eligible.csv ({len(bat_elig)} players)")
print(f"  Saved bowl_eligible.csv ({len(bowl_elig)} players)")

# ── STEP 4B – EVIDENCE DOSSIERS ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4B – EVIDENCE DOSSIERS")
print("=" * 70)

# Top-25 batters and bowlers for dossiers
TOP_N_DOSSIER = 30

bat_top  = bat_elig.nlargest(TOP_N_DOSSIER, 'wpae_bat_weighted')
bowl_top = bowl_elig.nlargest(TOP_N_DOSSIER, 'wpae_bowl_weighted')

finalist_pids = set(bat_top['player_id'].tolist() + bowl_top['player_id'].tolist())
print(f"  Generating dossiers for {len(finalist_pids)} finalists …")

# For each finalist, extract top-3 and worst-3 innings/spells
dossier_rows = []

for pid in finalist_pids:
    is_bat  = pid in bat_top['player_id'].values
    is_bowl = pid in bowl_top['player_id'].values
    
    if is_bat:
        inns = bat_mi[bat_mi['striker_id'] == pid].copy()
        inns['match_date'] = pd.to_datetime(inns['match_date'])
        # Merge match context
        if len(inns) > 0:
            top3 = inns.nlargest(3, 'wpae_bat_li_sum_capped')
            bot3 = inns.nsmallest(3, 'wpae_bat_li_sum_capped')
            for idx, row in pd.concat([top3, bot3]).iterrows():
                dossier_rows.append({
                    'player_id': pid,
                    'player_name': name_map.get(pid, str(pid)),
                    'role': 'bat',
                    'match_id': row['match_id'],
                    'match_date': row['match_date'],
                    'comp_name': row.get('comp_name', ''),
                    'innings': row['innings'],
                    'balls_faced': row['balls_faced'],
                    'runs_scored': row['runs_bat'],
                    'wpae': row['wpae_bat_li_sum_capped'],
                    'is_pressure_inn': row['is_pressure'],
                    'bat_role': row['bat_role'],
                    'scenario': ', '.join(['pressure' if row['is_pressure']>0.5 else 'ordinary']),
                })
    
    if is_bowl:
        spells = bowl_mi[bowl_mi['bowler_id'] == pid].copy()
        spells['match_date'] = pd.to_datetime(spells['match_date'])
        if len(spells) > 0:
            top3 = spells.nlargest(3, 'wpae_bowl_li_sum_capped')
            bot3 = spells.nsmallest(3, 'wpae_bowl_li_sum_capped')
            for idx, row in pd.concat([top3, bot3]).iterrows():
                dossier_rows.append({
                    'player_id': pid,
                    'player_name': name_map.get(pid, str(pid)),
                    'role': 'bowl',
                    'match_id': row['match_id'],
                    'match_date': row['match_date'],
                    'comp_name': row.get('comp_name', ''),
                    'innings': row['innings'],
                    'balls_bowled': row['balls_bowled'],
                    'runs_conceded': row['runs_conceded'],
                    'wickets_taken': row['wickets_taken'],
                    'wpae': row['wpae_bowl_li_sum_capped'],
                    'is_pressure_spell': row['is_pressure'],
                    'pace_or_spin': row['pace_or_spin'],
                    'scenario': ', '.join(['pressure' if row['is_pressure']>0.5 else 'ordinary']),
                })

dossier_df = pd.DataFrame(dossier_rows)
dossier_df.to_csv(RES / 'finalist_dossiers.csv', index=False)
print(f"  Saved finalist_dossiers.csv ({len(dossier_df):,} records)")

# Recent vs full-period chart (top 20 batters)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, df_pl, col_full, col_12, label in [
    (axes[0], bat_elig.nlargest(20, 'wpae_bat_weighted'), 'wpae_bat_weighted', 'wpae_last12', 'Batters'),
    (axes[1], bowl_elig.nlargest(20, 'wpae_bowl_weighted'), 'wpae_bowl_weighted', 'wpae_last12', 'Bowlers'),
]:
    names = [name_map.get(pid, str(pid))[:15] for pid in df_pl['player_id']]
    x = np.arange(len(names))
    ax.barh(x, df_pl[col_full].values, 0.4, label='Full period', color='steelblue', alpha=0.8)
    ax.barh(x + 0.4, df_pl[col_12].values, 0.4, label='Last 12m', color='darkorange', alpha=0.8)
    ax.set_yticks(x + 0.2); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Recency-Weighted WPAE'); ax.set_title(f'Full Period vs Last 12m — {label}')
    ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5)
fig.suptitle('Recent Form vs Full Period WPAE', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'recent_vs_full_period.png', dpi=150)
plt.close(fig)
print("  Saved recent_vs_full_period.png")

print("\n" + "=" * 70)
print("STEPS 4 + 4B COMPLETE")
print(f"  Eligible batters: {len(bat_elig)}")
print(f"  Eligible bowlers: {len(bowl_elig)}")
print("=" * 70)
