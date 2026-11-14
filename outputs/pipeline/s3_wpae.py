"""
s3_wpae.py – Unified WPAE Performance Measure + Peer-Group Context Models (Step 3)
SupeRR Selector S2 (v7)
"""
import warnings, pathlib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)

WORK  = pathlib.Path(r'D:\SupeRR_round_1')
CLEAN = WORK / 'outputs' / 'data_clean'
RES   = WORK / 'outputs' / 'results'
FIG   = WORK / 'outputs' / 'figures'

print("=" * 70)
print("STEP 3 – UNIFIED WPAE PERFORMANCE MEASURE")
print("=" * 70)

# ── Load pressure data ──────────────────────────────────────────────────────
print("\nLoading deliveries_pressure.parquet …")
df = pd.read_parquet(CLEAN / 'deliveries_pressure.parquet')
excl_df = pd.read_csv(RES / 'excluded_pakistan_players.csv')
PAK_IDS = set(excl_df['player_id'].tolist())
print(f"  Rows: {len(df):,}  WPA cols present: {all(c in df.columns for c in ['wpa_batter','wpa_bowler','li'])}")

# ── PEER GROUPS ──────────────────────────────────────────────────────────────
print("\n--- Defining Peer Groups ---")

# Convert categorical columns to string first
for col in ['pace_or_spin', 'phase', 'bat_role', 'batting_type']:
    if col in df.columns:
        df[col] = df[col].astype(str)

# BOWLER peer groups: pace_or_spin × phase (6 groups, possibly 12 if split by arm/style)
df['bowler_peer'] = df['pace_or_spin'].fillna('Unknown') + '_' + df['phase'].fillna('unknown')
bowler_peer_counts = df[df['is_legal']==1].groupby('bowler_peer')['bowler_id'].count()
print("  Bowler peer group sizes (legal balls):")
print(bowler_peer_counts.sort_values(ascending=False).to_string())

# BATTER peer groups: bat_role × batting_type (handedness) × pace_or_spin faced
df['batter_peer'] = (
    df['bat_role'].fillna('anchor') + '_' +
    df['batting_type'].fillna('RHB') + '_' +
    df['pace_or_spin'].fillna('Unknown')
)
batter_peer_counts = df[df['is_legal']==1].groupby('batter_peer')['striker_id'].count()
print("\n  Batter peer group sizes (legal balls):")
print(batter_peer_counts.sort_values(ascending=False).head(15).to_string())

# ── CONTEXT MODELS (Expected values per peer group × situation) ──────────────
print("\n--- Context Models ---")

legal = df[df['is_legal'] == 1].copy()

# Common feature builder
def build_ctx_features(sub, top_venues=20):
    sub = sub.copy()
    # Bin batter_balls_so_far: new(0-5), set(6-20), established(21+)
    sub['batter_state_bin'] = pd.cut(
        sub['batter_balls_so_far'].fillna(0),
        bins=[-1, 5, 20, 1000], labels=['new', 'set', 'estab']
    ).astype(str)
    # Bin dots_before
    sub['dots_bin'] = pd.cut(
        sub['dots_before'].fillna(0),
        bins=[-1, 0, 2, 100], labels=['0', '1-2', '3+']
    ).astype(str)
    # Top venues
    top_v = sub['venue_id'].value_counts().head(top_venues).index.tolist()
    sub['venue_grp'] = sub['venue_id'].where(sub['venue_id'].isin(top_v), other=-1).astype(str)
    # Scoring environment bin
    sub['scoring_bin'] = pd.cut(
        sub['match_rpo_per_ball'].fillna(sub['match_rpo_per_ball'].median()) * 6,
        bins=[0, 7.5, 9.5, 100], labels=['low', 'med', 'high']
    ).astype(str)
    return sub

# One-hot encode categoricals
def ohe(sub, cols):
    return pd.get_dummies(sub[cols], drop_first=True).astype(float)

# -- Bowler expected runs and wicket probability by peer group
print("  Fitting bowler context models …")
bowler_ctx_models = {}

for pg, grp in legal.groupby('bowler_peer'):
    if len(grp) < 300:
        continue
    grp = build_ctx_features(grp)
    feat_cols = ['over_0based', 'wickets_in_hand', 'is_ipl', 'scoring_bin',
                 'dots_bin', 'venue_grp']
    X_ohe = ohe(grp, feat_cols)
    X_num = grp[['over_0based', 'wickets_in_hand', 'is_ipl']].fillna(0).reset_index(drop=True)
    X = pd.concat([X_num, X_ohe.reset_index(drop=True)], axis=1).fillna(0)
    y_runs  = grp['runs_bat'].values.astype(float)
    y_wkts  = grp['is_wicket_bowl'].values.astype(float)
    
    pr = PoissonRegressor(alpha=1.0, max_iter=500)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    try:
        pr.fit(X, y_runs)
        lr.fit(X, y_wkts)
        bowler_ctx_models[pg] = {'poisson': pr, 'logistic': lr, 'feature_names': list(X.columns)}
    except Exception as e:
        print(f"    WARNING: bowler peer {pg} model failed: {e}")

print(f"  Bowler context models fitted for {len(bowler_ctx_models)} peer groups")

# -- Batter expected runs and wicket probability by peer group
print("  Fitting batter context models …")
batter_ctx_models = {}

for pg, grp in legal.groupby('batter_peer'):
    if len(grp) < 300:
        continue
    grp = build_ctx_features(grp)
    feat_cols = ['over_0based', 'wickets_in_hand', 'is_ipl', 'scoring_bin',
                 'batter_state_bin', 'dots_bin', 'venue_grp']
    X_ohe = ohe(grp, feat_cols)
    X_num = grp[['over_0based', 'wickets_in_hand', 'is_ipl']].fillna(0).reset_index(drop=True)
    X = pd.concat([X_num, X_ohe.reset_index(drop=True)], axis=1).fillna(0)
    y_runs = grp['runs_bat'].values.astype(float)
    y_wkts = grp['is_wicket_bat'].values.astype(float)
    
    pr = PoissonRegressor(alpha=1.0, max_iter=500)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    try:
        pr.fit(X, y_runs)
        lr.fit(X, y_wkts)
        batter_ctx_models[pg] = {'poisson': pr, 'logistic': lr, 'feature_names': list(X.columns)}
    except Exception as e:
        print(f"    WARNING: batter peer {pg} model failed: {e}")

print(f"  Batter context models fitted for {len(batter_ctx_models)} peer groups")

# ── COMPUTE EXPECTED WPA (E[WPA]) ─────────────────────────────────────────────
# Instead of per-ball context model WPA (complex), we use:
# E[wpa_batter] = peer-group mean WPA for that ball's situation (binned)
# This is equivalent to computing peer-group adjusted WPA
print("\n--- Computing Expected WPA (peer-group adjusted) ---")

# Compute peer-group-situation mean WPA for bowlers
# Situation bins: phase × wickets_in_hand bucket × scoring_bin
legal['wih_bin'] = pd.cut(legal['wickets_in_hand'].fillna(5), bins=[-1,2,5,10],
                           labels=['low','mid','high']).astype(str)
legal['scoring_bin2'] = pd.cut(legal['match_rpo_per_ball'].fillna(legal['match_rpo_per_ball'].median())*6,
                                bins=[0,7.5,9.5,100], labels=['low','med','high']).astype(str)
legal['situation_key'] = legal['phase'] + '_' + legal['wih_bin'] + '_' + legal['scoring_bin2']

# Bowler expected WPA
bowl_peer_sit = legal.groupby(['bowler_peer', 'situation_key'])['wpa_bowler'].mean().reset_index()
bowl_peer_sit.columns = ['bowler_peer', 'situation_key', 'e_wpa_bowler']

# Batter expected WPA  
bat_peer_sit = legal.groupby(['batter_peer', 'situation_key'])['wpa_batter'].mean().reset_index()
bat_peer_sit.columns = ['batter_peer', 'situation_key', 'e_wpa_batter']

# Merge expectations back
legal = legal.merge(bowl_peer_sit, on=['bowler_peer', 'situation_key'], how='left')
legal = legal.merge(bat_peer_sit,  on=['batter_peer',  'situation_key'], how='left')

# Fill missing with peer-group mean
peer_bowl_mean = legal.groupby('bowler_peer')['wpa_bowler'].transform('mean')
peer_bat_mean  = legal.groupby('batter_peer') ['wpa_batter'].transform('mean')
legal['e_wpa_bowler'] = legal['e_wpa_bowler'].fillna(peer_bowl_mean)
legal['e_wpa_batter'] = legal['e_wpa_batter'].fillna(peer_bat_mean)
legal['e_wpa_bowler'] = legal['e_wpa_bowler'].fillna(0)
legal['e_wpa_batter'] = legal['e_wpa_batter'].fillna(0)

# WPAE (Win Probability Added above Expected)
legal['wpae_bat']  = legal['wpa_batter'] - legal['e_wpa_batter']
legal['wpae_bowl'] = legal['wpa_bowler'] - legal['e_wpa_bowler']

# Extra weight for rescue from unfavourable position (WP < 40%)
rescue_mask = legal['wp_before'] < 0.40
legal['wpae_bat_w']  = legal['wpae_bat']  * (1 + 0.3 * rescue_mask.astype(float))
legal['wpae_bowl_w'] = legal['wpae_bowl'] * (1 + 0.3 * rescue_mask.astype(float))

# LI-weighted WPAE (additional emphasis on high-leverage moments)
legal['wpae_bat_li']  = legal['wpae_bat_w']  * np.sqrt(legal['li'].clip(0.1, 5))
legal['wpae_bowl_li'] = legal['wpae_bowl_w'] * np.sqrt(legal['li'].clip(0.1, 5))

print(f"  WPAE bat: mean={legal.wpae_bat.mean():.5f}  std={legal.wpae_bat.std():.4f}")
print(f"  WPAE bowl: mean={legal.wpae_bowl.mean():.5f}  std={legal.wpae_bowl.std():.4f}")

# ── CAP SINGLE EVENT CONTRIBUTION ────────────────────────────────────────────
# Cap so one innings/spell can't decide ranking: cap at 99th percentile per match
print("\n--- Capping single-event contributions ---")
match_inn_bat  = legal.groupby(['match_id', 'innings', 'striker_id'])['wpae_bat_li'].sum()
match_inn_bowl = legal.groupby(['match_id', 'innings', 'bowler_id'])['wpae_bowl_li'].sum()

cap_bat  = match_inn_bat.quantile(0.99)
cap_bowl = match_inn_bowl.quantile(0.99)
print(f"  Cap (99th pct): bat={cap_bat:.4f}  bowl={cap_bowl:.4f}")

# Note: capping is done at aggregation stage (Step 4), not per-ball

# ── AGGREGATE PLAYER RATINGS ─────────────────────────────────────────────────
print("\n--- Aggregating raw player ratings ---")

# BATTER ratings (per match-innings, then sum)
bat_mi = (
    legal.groupby(['striker_id', 'match_id', 'innings'])
    .agg(
        balls_faced      = ('wpae_bat_li', 'count'),
        wpae_bat_li_sum  = ('wpae_bat_li', 'sum'),
        wpae_bat_sum     = ('wpae_bat',    'sum'),
        wpa_bat_sum      = ('wpa_batter',  'sum'),
        runs_bat         = ('runs_bat',    'sum'),
        is_wicket        = ('is_wicket_bat','sum'),
        is_pressure      = ('is_pressure_ball','mean'),
        phase            = ('phase',       lambda x: x.mode().iloc[0] if len(x)>0 else 'middle'),
        match_date       = ('match_date',  'first'),
        comp_name        = ('comp_name',   'first'),
        bat_role         = ('bat_role',    lambda x: x.mode().iloc[0] if len(x)>0 else 'anchor'),
    ).reset_index()
)

# Cap per-innings bat contribution
bat_mi['wpae_bat_li_sum_capped'] = bat_mi['wpae_bat_li_sum'].clip(upper=cap_bat)

# BOWLER ratings (per match-innings)
bowl_mi = (
    legal.groupby(['bowler_id', 'match_id', 'innings'])
    .agg(
        balls_bowled      = ('wpae_bowl_li', 'count'),
        wpae_bowl_li_sum  = ('wpae_bowl_li', 'sum'),
        wpae_bowl_sum     = ('wpae_bowl',    'sum'),
        wpa_bowl_sum      = ('wpa_bowler',   'sum'),
        runs_conceded     = ('total_del_runs','sum'),
        wickets_taken     = ('is_wicket_bowl','sum'),
        is_pressure       = ('is_pressure_ball','mean'),
        phase             = ('phase',        lambda x: x.mode().iloc[0] if len(x)>0 else 'middle'),
        match_date        = ('match_date',   'first'),
        comp_name         = ('comp_name',    'first'),
        pace_or_spin      = ('pace_or_spin', lambda x: x.mode().iloc[0] if len(x)>0 else 'Pace'),
        bowling_type      = ('bowling_type', lambda x: x.mode().iloc[0] if len(x)>0 else 'Unknown'),
    ).reset_index()
)
bowl_mi['wpae_bowl_li_sum_capped'] = bowl_mi['wpae_bowl_li_sum'].clip(upper=cap_bowl)

print(f"  Batter match-innings records: {len(bat_mi):,}")
print(f"  Bowler match-innings records: {len(bowl_mi):,}")

# Total career aggregates (pre-recency; recency done in Step 4)
bat_career = (
    bat_mi.groupby('striker_id')
    .agg(
        total_balls      = ('balls_faced',           'sum'),
        total_wpae_li    = ('wpae_bat_li_sum_capped','sum'),
        total_wpae       = ('wpae_bat_sum',          'sum'),
        total_wpa        = ('wpa_bat_sum',           'sum'),
        n_innings        = ('match_id',              'nunique'),
        total_runs       = ('runs_bat',              'sum'),
        total_wickets    = ('is_wicket',             'sum'),
        mean_pressure    = ('is_pressure',           'mean'),
        bat_role         = ('bat_role',              lambda x: x.mode().iloc[0] if len(x)>0 else 'anchor'),
        last_match_date  = ('match_date',            'max'),
        n_comps          = ('comp_name',             'nunique'),
    ).reset_index().rename(columns={'striker_id': 'player_id'})
)
bat_career['sr'] = 100 * bat_career['total_runs'] / bat_career['total_balls'].clip(lower=1)
bat_career['dismissal_rate'] = bat_career['total_wickets'] / bat_career['n_innings'].clip(lower=1)

bowl_career = (
    bowl_mi.groupby('bowler_id')
    .agg(
        total_balls      = ('balls_bowled',          'sum'),
        total_wpae_li    = ('wpae_bowl_li_sum_capped','sum'),
        total_wpae       = ('wpae_bowl_sum',         'sum'),
        total_wpa        = ('wpa_bowl_sum',          'sum'),
        n_spells         = ('match_id',              'nunique'),
        total_runs_conc  = ('runs_conceded',         'sum'),
        total_wickets    = ('wickets_taken',         'sum'),
        mean_pressure    = ('is_pressure',           'mean'),
        pace_or_spin     = ('pace_or_spin',          lambda x: x.mode().iloc[0] if len(x)>0 else 'Pace'),
        bowling_type     = ('bowling_type',          lambda x: x.mode().iloc[0] if len(x)>0 else 'Unknown'),
        last_match_date  = ('match_date',            'max'),
        n_comps          = ('comp_name',             'nunique'),
    ).reset_index().rename(columns={'bowler_id': 'player_id'})
)
bowl_career['economy'] = 6 * bowl_career['total_runs_conc'] / bowl_career['total_balls'].clip(lower=1)
bowl_career['wickets_per_match'] = bowl_career['total_wickets'] / bowl_career['n_spells'].clip(lower=1)

print(f"\n  Bat career: {len(bat_career)} players")
print(f"  Bowl career: {len(bowl_career)} players")
print(f"  Top bat WPAE: {bat_career.nlargest(5,'total_wpae_li')[['player_id','total_balls','total_wpae_li']].to_string()}")
print(f"  Top bowl WPAE: {bowl_career.nlargest(5,'total_wpae_li')[['player_id','total_balls','total_wpae_li']].to_string()}")

# ── BOWLER SKILL PROFILES ─────────────────────────────────────────────────────
print("\n--- Bowler Skill Profiles ---")

bowl_prof = (
    legal.groupby(['bowler_id', 'phase'])
    .agg(
        balls            = ('is_legal',      'sum'),
        runs_conceded    = ('total_del_runs','sum'),
        wickets          = ('is_wicket_bowl','sum'),
        dots             = ('runs_bat',      lambda x: (x==0).sum()),
        boundaries       = ('runs_bat',      lambda x: (x>=4).sum()),
        sixes            = ('is_six',        'sum'),
        yorkers          = ('is_yorker',     'sum'),
        bouncers         = ('is_bouncer',    'sum'),
        mean_speed       = ('bowl_speed',    'mean'),
        wpae_bowl        = ('wpae_bowl_li',  'sum'),
    ).reset_index()
)
bowl_prof['economy']   = 6 * bowl_prof['runs_conceded'] / bowl_prof['balls'].clip(lower=1)
bowl_prof['dot_pct']   = 100 * bowl_prof['dots'] / bowl_prof['balls'].clip(lower=1)
bowl_prof['bdry_pct']  = 100 * bowl_prof['boundaries'] / bowl_prof['balls'].clip(lower=1)
bowl_prof['six_pct']   = 100 * bowl_prof['sixes'] / bowl_prof['balls'].clip(lower=1)
bowl_prof['wkts_per10']= 10 * bowl_prof['wickets'] / bowl_prof['balls'].clip(lower=1)
bowl_prof['yorker_pct']= 100 * bowl_prof['yorkers'] / bowl_prof['balls'].clip(lower=1)
bowl_prof['bounce_pct']= 100 * bowl_prof['bouncers'] / bowl_prof['balls'].clip(lower=1)

# Peer-group expected values for each metric
pg_prof = (
    legal.groupby(['bowler_peer', 'phase'])
    .agg(
        balls_pg   = ('is_legal',      'sum'),
        eco_pg_num = ('total_del_runs','sum'),
        dot_pg_num = ('runs_bat',      lambda x: (x==0).sum()),
        bdry_pg_num= ('runs_bat',      lambda x: (x>=4).sum()),
        wkt_pg_num = ('is_wicket_bowl','sum'),
    ).reset_index()
)
pg_prof['eco_pg']  = 6 * pg_prof['eco_pg_num'] / pg_prof['balls_pg'].clip(lower=1)
pg_prof['dot_pct_pg'] = 100 * pg_prof['dot_pg_num'] / pg_prof['balls_pg'].clip(lower=1)
pg_prof['bdry_pct_pg']= 100 * pg_prof['bdry_pg_num']/ pg_prof['balls_pg'].clip(lower=1)
pg_prof['wkts_per10_pg']= 10 * pg_prof['wkt_pg_num'] / pg_prof['balls_pg'].clip(lower=1)

# Join bowler peer to deliveries to get peer-group expectations
# Bowler → primary pace_or_spin
bowler_ps = legal.groupby('bowler_id')['pace_or_spin'].agg(lambda x: x.mode().iloc[0] if len(x)>0 else 'Unknown')
bowl_prof = bowl_prof.merge(bowler_ps.reset_index().rename(columns={'bowler_id':'bowler_id','pace_or_spin':'pace_or_spin'}),
                             on='bowler_id', how='left')
bowl_prof['bowler_peer'] = bowl_prof['pace_or_spin'].fillna('Unknown') + '_' + bowl_prof['phase'].fillna('unknown')
bowl_prof = bowl_prof.merge(pg_prof[['bowler_peer','phase','eco_pg','dot_pct_pg','bdry_pct_pg','wkts_per10_pg']],
                             on=['bowler_peer','phase'], how='left')
bowl_prof['eco_vs_pg']  = bowl_prof['economy']   - bowl_prof['eco_pg']
bowl_prof['dot_vs_pg']  = bowl_prof['dot_pct']   - bowl_prof['dot_pct_pg']
bowl_prof['bdry_vs_pg'] = bowl_prof['bdry_pct']  - bowl_prof['bdry_pct_pg']
bowl_prof['wkt_vs_pg']  = bowl_prof['wkts_per10']- bowl_prof['wkts_per10_pg']

print(f"  Bowl skill profile: {len(bowl_prof):,} rows (player × phase)")

# ── BATTER SKILL PROFILES ─────────────────────────────────────────────────────
print("\n--- Batter Skill Profiles ---")

bat_prof = (
    legal.groupby(['striker_id', 'phase'])
    .agg(
        balls_faced = ('is_legal',       'sum'),
        runs_scored = ('runs_bat',       'sum'),
        wickets     = ('is_wicket_bat',  'sum'),
        dots        = ('runs_bat',       lambda x: (x==0).sum()),
        fours       = ('is_four',        'sum'),
        sixes       = ('is_six',         'sum'),
        wpae_bat    = ('wpae_bat_li',    'sum'),
        mean_li     = ('li',             'mean'),
        is_pressure = ('is_pressure_ball','mean'),
    ).reset_index()
)
bat_prof['sr']       = 100 * bat_prof['runs_scored'] / bat_prof['balls_faced'].clip(lower=1)
bat_prof['dot_pct']  = 100 * bat_prof['dots']  / bat_prof['balls_faced'].clip(lower=1)
bat_prof['bdry_pct'] = 100 * (bat_prof['fours'] + bat_prof['sixes']) / bat_prof['balls_faced'].clip(lower=1)
bat_prof['six_pct']  = 100 * bat_prof['sixes'] / bat_prof['balls_faced'].clip(lower=1)
bat_prof['dismissal_rate'] = bat_prof['wickets'] / bat_prof['balls_faced'].clip(lower=1)

# Peer-group expected SR
# batter peer uses bat_role + batting_type + pace_or_spin
bat_peer_phase = (
    legal.groupby(['batter_peer', 'phase'])
    .agg(
        sr_pg_num  = ('runs_bat', 'sum'),
        sr_pg_den  = ('is_legal', 'sum'),
        dot_pg_num = ('runs_bat', lambda x: (x==0).sum()),
    ).reset_index()
)
bat_peer_phase['sr_pg']     = 100 * bat_peer_phase['sr_pg_num'] / bat_peer_phase['sr_pg_den'].clip(lower=1)
bat_peer_phase['dot_pct_pg']= 100 * bat_peer_phase['dot_pg_num'] / bat_peer_phase['sr_pg_den'].clip(lower=1)

# Join batter peer to bat_prof
batter_peer_map = legal.groupby('striker_id')['batter_peer'].agg(lambda x: x.mode().iloc[0] if len(x)>0 else 'anchor_RHB_Pace')
bat_prof = bat_prof.merge(batter_peer_map.reset_index().rename(columns={'striker_id':'striker_id','batter_peer':'batter_peer'}),
                           on='striker_id', how='left')
bat_prof = bat_prof.merge(bat_peer_phase[['batter_peer','phase','sr_pg','dot_pct_pg']],
                           on=['batter_peer','phase'], how='left')
bat_prof['sr_vs_pg']  = bat_prof['sr']      - bat_prof['sr_pg']
bat_prof['dot_vs_pg'] = bat_prof['dot_pct'] - bat_prof['dot_pct_pg']

print(f"  Bat skill profile: {len(bat_prof):,} rows (player × phase)")

# ── SAVE INTERMEDIATE RESULTS ─────────────────────────────────────────────────
print("\n--- Saving intermediate files ---")
legal.to_parquet(CLEAN / 'deliveries_wpae.parquet', index=False)
bat_career.to_csv(RES / 'bat_career_raw.csv', index=False)
bowl_career.to_csv(RES / 'bowl_career_raw.csv', index=False)
bat_mi.to_parquet(CLEAN / 'bat_match_innings.parquet', index=False)
bowl_mi.to_parquet(CLEAN / 'bowl_match_innings.parquet', index=False)
bowl_prof.to_csv(RES / 'bowl_skill_profiles.csv', index=False)
bat_prof.to_csv(RES / 'bat_skill_profiles.csv', index=False)

print(f"  deliveries_wpae.parquet  ({len(legal):,} rows)")
print(f"  bat_career_raw.csv       ({len(bat_career):,} players)")
print(f"  bowl_career_raw.csv      ({len(bowl_career):,} players)")
print(f"  bat_match_innings.parquet ({len(bat_mi):,} records)")
print(f"  bowl_match_innings.parquet ({len(bowl_mi):,} records)")
print(f"  bowl_skill_profiles.csv  ({len(bowl_prof):,} rows)")
print(f"  bat_skill_profiles.csv   ({len(bat_prof):,} rows)")

print("\n" + "=" * 70)
print("STEP 3 COMPLETE")
print("=" * 70)
