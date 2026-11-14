"""
SupeRR Selector S2 v7 — Step 0 + Step 1
Scope audit, data dictionary, Pakistan exclusion, clean delivery table.
RANDOM_SEED = 42
"""
import os, warnings, json
warnings.filterwarnings('ignore')
os.environ['MPLCONFIGDIR'] = r'C:\Users\ajaym\AppData\Local\Temp\mpl_cfg'
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED    = 42
WS      = r'D:\SupeRR_round_1'
OUT     = os.path.join(WS, 'outputs')
DATA    = os.path.join(OUT, 'data_clean')
RES     = os.path.join(OUT, 'results')
FIG     = os.path.join(OUT, 'figures')
for d in [DATA, RES, FIG]: os.makedirs(d, exist_ok=True)

np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════
# STEP 0 — SCOPE AUDIT & PAKISTAN EXCLUSION
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 0 — SCOPE AUDIT & PAKISTAN EXCLUSION")
print("=" * 65)

# Load reference files
comp = pd.read_csv(os.path.join(WS, 'competitions.csv'))
mat  = pd.read_csv(os.path.join(WS, 'matches.csv'))
pe   = pd.read_csv(os.path.join(WS, 'playing_eleven.csv'))
teams= pd.read_csv(os.path.join(WS, 'teams.csv'))
ctpm = pd.read_csv(os.path.join(WS, 'competition_team_player_mapping.csv'))
bbb  = pd.read_csv(os.path.join(WS, 'ball_by_ball.csv'), low_memory=False)

print(f"\nRaw files loaded:")
print(f"  competitions : {len(comp):6,} rows")
print(f"  matches      : {len(mat):6,} rows")
print(f"  playing_eleven: {len(pe):6,} rows")
print(f"  teams        : {len(teams):6,} rows")
print(f"  ctpm         : {len(ctpm):6,} rows")
print(f"  ball_by_ball : {len(bbb):6,} rows")

# ── 0A. Data dictionary ──────────────────────────────────────
print("\n=== BALL-BY-BALL DATA DICTIONARY ===")
dd = []
for col in bbb.columns:
    s     = bbb[col]
    nulls = s.isna().sum()
    dtype = str(s.dtype)
    nuniq = s.nunique(dropna=True)
    sample_vals = s.dropna().unique()[:4]
    dd.append({'column': col, 'dtype': dtype, 'nulls': nulls,
               'n_unique': nuniq, 'sample': str(list(sample_vals))})
    print(f"  {col:30s}  {dtype:10s}  nulls={nulls:6,}  unique={nuniq:5,}  sample={list(sample_vals)[:3]}")

pd.DataFrame(dd).to_csv(os.path.join(RES, 'data_dictionary.csv'), index=False)

# ── Speed / Line / Length check ──────────────────────────────
print("\n=== SPECIAL COLUMNS CHECK ===")
spd = bbb[bbb['bowl_speed'] > 0]['bowl_speed']
print(f"  bowl_speed  : FOUND — {len(spd):,} non-zero values, "
      f"min={spd.min():.1f} max={spd.max():.1f} mean={spd.mean():.1f} km/h")
print(f"  delivery_style: FOUND — {bbb['delivery_style'].nunique()} unique delivery types")
print(f"    {bbb['delivery_style'].value_counts().head(8).to_dict()}")
print(f"  stroke       : FOUND — {bbb['stroke'].nunique()} unique shot types")
print(f"  over_or_around: FOUND — {bbb['over_or_around'].value_counts().to_dict()}")
print(f"  on_or_off    : FOUND — {bbb['on_or_off'].value_counts().to_dict()}")
print(f"  line/length  : NO dedicated line/length columns — using delivery_style as proxy")

# ── 0B. Scope filtering ──────────────────────────────────────
print("\n=== 0B. SCOPE FILTERING ===")
print(f"Before filter: {len(mat):,} matches, {len(comp):,} competitions")

# Keep IPL and T20I; drop A-team events
DROP_COMPS = ['Mens T20 Emerging Teams Asia Cup', 'Asia Cup Rising Stars',
              'Asia Cup T20 Qualifiers']
def is_drop(name):
    for d in DROP_COMPS:
        if d.lower() in str(name).lower():
            return True
    return False

comp_keep = comp[
    (comp['comp_name'].str.startswith('IPL') |
     (comp['match_format'] == 'T20I')) &
    ~comp['comp_name'].apply(is_drop)
].copy()

mat['match_date'] = pd.to_datetime(mat['match_date'], errors='coerce')
comp_keep['season_year'] = comp_keep['season'].str[:4].astype(int)
comp_keep = comp_keep[comp_keep['season_year'] >= 2022]

mat_scoped = mat[mat['comp_id'].isin(comp_keep['comp_id'])].copy()
print(f"After IPL+T20I+2022+ filter: {len(mat_scoped):,} matches")

# Drop No Result and Abandoned
mat_valid = mat_scoped[
    ~mat_scoped['result'].str.contains('No Result|Abandoned', na=False, case=False)
].copy()
print(f"After dropping No Result/Abandoned: {len(mat_valid):,} matches")

# Format breakdown
ipl_m   = mat_valid[mat_valid['comp_id'].isin(
    comp_keep[comp_keep['comp_name'].str.startswith('IPL')]['comp_id'])]['match_id'].nunique()
t20i_m  = mat_valid[mat_valid['comp_id'].isin(
    comp_keep[comp_keep['match_format'] == 'T20I']['comp_id'])]['match_id'].nunique()
print(f"  IPL matches  : {ipl_m}")
print(f"  T20I matches : {t20i_m}")

# Join match metadata to competition
mat_valid = mat_valid.merge(
    comp_keep[['comp_id', 'comp_name', 'match_format', 'season', 'season_year']],
    on='comp_id', how='left')
mat_valid['is_ipl'] = mat_valid['comp_name'].str.startswith('IPL').fillna(False).astype(int)

# Valid match ids
valid_mids = set(mat_valid['match_id'].tolist())

# ── 0C. Pakistan exclusion ───────────────────────────────────
print("\n=== 0C. PAKISTAN EXCLUSION ===")
PAK_TEAM_IDS = [25, 341]
pak_ctpm = ctpm[ctpm['team_name'].str.contains('Pakistan', case=False, na=False)]
pak_pe   = pe[pe['team_id'].isin(PAK_TEAM_IDS)]
all_pak_ids = set(pak_ctpm['player_id'].tolist()) | set(pak_pe['player_id'].tolist())
name_map = ctpm.groupby('player_id')['player_name'].agg(
    lambda x: x.value_counts().index[0])

excl_rows = []
for pid in sorted(all_pak_ids):
    n   = name_map.get(pid, f'ID_{pid}')
    ctm = pak_ctpm[pak_ctpm['player_id'] == pid]
    te  = ctm['team_name'].iloc[0] if len(ctm) > 0 else 'Pakistan (pe only)'
    pem = pid in set(pak_pe['player_id'])
    reasons = []
    if len(ctm) > 0: reasons.append(f'ctpm:team={te}')
    if pem:          reasons.append('playing_eleven:team_id=25/341')
    excl_rows.append({'player_id': pid, 'player_name': n, 'reason': '; '.join(reasons)})

excl_df = pd.DataFrame(excl_rows)
excl_df.to_csv(os.path.join(RES, 'excluded_pakistan_players.csv'), index=False)
print(f"Pakistan player_ids excluded: {len(all_pak_ids)}")
print(excl_df[['player_id', 'player_name']].to_string())

# ── 0D. BBB scope filter (keep all deliveries for valid matches) ──
print("\n=== 0D. BBB SCOPE ===")
bbb_all = bbb[bbb['match_id'].isin(valid_mids)].copy()
print(f"BBB rows in scope: {len(bbb_all):,}")
# Super overs — keep them tagged but separate from main model
bbb_all['is_super_over'] = bbb_all['innings'].isin([3, 4]).astype(int)
print(f"  Super-over deliveries: {bbb_all['is_super_over'].sum():,}")
bbb_main = bbb_all[bbb_all['is_super_over'] == 0].copy()
print(f"  Main-innings deliveries: {len(bbb_main):,}")
# Check coverage
bbb_mids = set(bbb_main['match_id'].unique())
print(f"  Matches with BBB data: {len(bbb_mids):,} of {len(valid_mids):,}")
missing  = valid_mids - bbb_mids
print(f"  Matches WITHOUT BBB: {len(missing)}")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — CLEAN DELIVERY TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1 — BUILDING CLEAN DELIVERY TABLE")
print("=" * 65)

df = bbb_main.copy()

# ── 1A. Extra classification ─────────────────────────────────
# extras_type: NaN=legal, 'wide', 'Overstep'=noball, 'byes', 'legbyes', 'other','Penalty'
df['is_wide']     = (df['extras_type'] == 'wide').astype(int)
df['is_noball']   = (df['extras_type'] == 'Overstep').astype(int)
df['is_bye']      = (df['extras_type'] == 'byes').astype(int)
df['is_legbye']   = (df['extras_type'] == 'legbyes').astype(int)
df['is_penalty']  = (df['extras_type'] == 'Penalty').astype(int)
df['is_legal']    = ((df['is_wide'] == 0) & (df['is_noball'] == 0)).astype(int)

# Runs off bat (excludes extras)
df['runs_bat']    = df['runs'].where(df['is_bye'] == 0, 0)
df['runs_bat']    = df['runs_bat'].where(df['is_legbye'] == 0, 0)
df['runs_bat']    = df['runs_bat'].where(df['is_wide'] == 0, 0)
# runs_bat = runs scored off the bat (not extras)
df['runs_bat']    = np.where(df['extras_type'].isna(), df['runs'], 0)
df['extras_runs'] = df['extras1_runs'] + df['extras_2_runs']
df['total_del_runs'] = df['runs'] + df['extras_runs']

# Wicket on this ball (not run-out for bowler)
df['is_wicket']   = df['is_wicket'].astype(int)
# Dismissal type
df['dismissal_type'] = df['dismissal_type'].astype(str).str.lower().str.strip()
df['is_runout']   = df['dismissal_type'].str.contains('run out', na=False).astype(int)
df['is_wicket_bat']  = (df['is_wicket'] == 1).astype(int)         # counts for batter
df['is_wicket_bowl'] = ((df['is_wicket'] == 1) & (df['is_runout'] == 0)).astype(int) # counts for bowler

# Dismissed player
df['dismissed_player_id'] = pd.to_numeric(df['dismissed_player_id'], errors='coerce')

# ── 1B. Bowl speed — clean outliers ─────────────────────────
# Valid range: 60-180 km/h for pace, 40-120 for spin
df['bowl_speed_raw'] = df['bowl_speed']
df['bowl_speed'] = df['bowl_speed_raw'].where(
    (df['bowl_speed_raw'] >= 50) & (df['bowl_speed_raw'] <= 180), np.nan)
spd_ok = df['bowl_speed'].notna().sum()
print(f"bowl_speed valid values after cleaning: {spd_ok:,} "
      f"({100*spd_ok/len(df):.1f}% of deliveries)")

# ── 1C. Join match context ───────────────────────────────────
mat_ctx = mat_valid[['match_id','comp_id','comp_name','match_format','season','season_year',
                      'match_date','stage','venue_id','venue_country','is_ipl',
                      'knock_out','win_team_id','result','revised_target',
                      'first_inn_revised_overs','second_inn_revised_overs',
                      'home_team_id','away_team_id']].copy()
df = df.merge(mat_ctx, on='match_id', how='left')
df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')

# ── 1D. Phase ────────────────────────────────────────────────
df['over_0based'] = df['overid'] - 1   # 0–19
df['phase'] = pd.cut(df['over_0based'], bins=[-1, 5, 14, 19],
                     labels=['powerplay', 'middle', 'death'])

# ── 1E. Knockout / stage flag ────────────────────────────────
ko_stages = ['final','qualifier','eliminator','semi final','super 4']
df['is_knockout'] = (
    (df['knock_out'] == 1) |
    df['stage'].str.lower().str.strip().isin(ko_stages)
).astype(int)

# ── 1F. Delivery style features ─────────────────────────────
# Yorker proxy: delivery_style == 'Yorker'
df['is_yorker']    = (df['delivery_style'] == 'Yorker').astype(int)
df['is_bouncer']   = (df['delivery_style'] == 'Bouncer').astype(int)
df['is_full_toss'] = (df['delivery_style'] == 'Full Toss').astype(int)

# ── 1G. Compute in-innings running state (legal balls only) ──
print("Computing in-innings running states...")
df = df.sort_values(['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Legal-ball cumulative count per innings
df['legal_ball_seq'] = df.groupby(['match_id','innings'])['is_legal'].cumsum()
# Runs so far (batting team) — use total_del_runs
df['team_runs_so_far'] = df.groupby(['match_id','innings'])['total_del_runs'].cumsum() - df['total_del_runs']
# Wickets lost so far
df['wickets_so_far'] = df.groupby(['match_id','innings'])['is_wicket_bat'].cumsum() - df['is_wicket_bat']
df['wickets_in_hand'] = 10 - df['wickets_so_far']

# Total overs (use revised if DLS)
df['total_overs'] = np.where(
    df['innings'] == 2,
    df['second_inn_revised_overs'].fillna(20),
    df['first_inn_revised_overs'].fillna(20)
)
df['legal_balls_total'] = (df['total_overs'] * 6).astype(int)
df['legal_balls_bowled'] = df['legal_ball_seq']
df['legal_balls_remaining'] = (df['legal_balls_total'] - df['legal_balls_bowled']).clip(lower=0)

# CRR and RRR (innings 2 only meaningful for RRR)
df['crr'] = np.where(
    df['legal_balls_bowled'] > 0,
    df['team_runs_so_far'] / (df['legal_balls_bowled'] / 6.0),
    0.0)
df['target'] = np.where(
    df['innings'] == 2,
    df['revised_target'].where(df['revised_target'].notna(),
        df.groupby('match_id')['team_runs_so_far'].transform('max') + 1),
    np.nan)
df['runs_required'] = np.where(
    df['innings'] == 2,
    (df['target'] - df['team_runs_so_far']).clip(lower=0),
    np.nan)
df['rrr'] = np.where(
    (df['innings'] == 2) & (df['legal_balls_remaining'] > 0),
    df['runs_required'] / (df['legal_balls_remaining'] / 6.0),
    np.nan)
df['rrr_minus_crr'] = df['rrr'] - df['crr']

# ── 1H. Batter state (balls faced this innings) ──────────────
# Count legal balls faced by striker within this match-innings
df['batter_balls_so_far'] = df[df['is_legal'] == 1].groupby(
    ['match_id','innings','striker_id']).cumcount()
# Fill NaN for non-legal balls
df['batter_balls_so_far'] = df.groupby(
    ['match_id','innings','striker_id'])['batter_balls_so_far'].transform(
    lambda x: x.ffill().fillna(0))
df['batter_is_new'] = (df['batter_balls_so_far'] < 10).astype(int)
df['batter_is_set']  = (df['batter_balls_so_far'] >= 20).astype(int)

# ── 1I. Consecutive dots before this ball ───────────────────
# Within match-innings, count consecutive dots (runs_bat==0 on legal balls)
legal_df = df[df['is_legal'] == 1].copy()
legal_df['is_dot'] = (legal_df['runs_bat'] == 0).astype(int)

def consec_dots(arr):
    result = np.zeros(len(arr), dtype=int)
    count  = 0
    for i, v in enumerate(arr):
        result[i] = count
        count = count + 1 if v == 1 else 0
    return result

legal_df['dots_before'] = legal_df.groupby(['match_id','innings'])['is_dot'].transform(
    lambda x: consec_dots(x.values))
df = df.merge(legal_df[['match_id','innings','overid','over_ball_num','dots_before']],
              on=['match_id','innings','overid','over_ball_num'], how='left')
df['dots_before'] = df['dots_before'].fillna(0).astype(int)

# ── 1J. Bowler over count ────────────────────────────────────
df['bowler_overs_so_far'] = df[df['is_legal'] == 1].groupby(
    ['match_id','innings','bowler_id'])['overid'].transform(
    lambda x: x.rank(method='dense') - 1)
df['bowler_overs_so_far'] = df['bowler_overs_so_far'].fillna(0)

# ── 1K. First-innings par score (venue+season) ───────────────
print("Computing venue par scores...")
inn1 = df[(df['innings'] == 1) & (df['is_legal'] == 1)].groupby(
    ['match_id','venue_id','season_year']).agg(
    total_runs=('total_del_runs','sum')).reset_index()
venue_par = inn1.groupby(['venue_id','season_year'])['total_runs'].median().reset_index()
venue_par.columns = ['venue_id','season_year','venue_par']
df = df.merge(venue_par, on=['venue_id','season_year'], how='left')
# Fill with competition median
comp_par = inn1.groupby('season_year')['total_runs'].median()
df['venue_par'] = df['venue_par'].fillna(df['season_year'].map(comp_par))
df['venue_par'] = df['venue_par'].fillna(160)  # global fallback

# Match scoring environment (runs per over of the match)
match_runs = df[df['is_legal'] == 1].groupby('match_id')['total_del_runs'].mean().reset_index()
match_runs.columns = ['match_id','match_rpo_per_ball']
df = df.merge(match_runs, on='match_id', how='left')

# ── 1L. Batting role from playing_eleven ────────────────────
pe_role = pe[pe['is_played'] == 1][['match_id','player_id','batting_order']].copy()
pe_role['bat_role'] = pd.cut(pe_role['batting_order'],
    bins=[0, 2, 5, 11], labels=['opener','anchor','finisher'])
def safe_mode(x):
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else 'anchor'
batter_role_map = pe_role.groupby('player_id')['bat_role'].agg(safe_mode)
df['bat_role'] = df['striker_id'].map(batter_role_map).fillna('anchor')

# ── 1M. Final clean table ────────────────────────────────────
print(f"\nFinal clean table: {len(df):,} rows × {len(df.columns)} cols")
print(f"Legal deliveries: {df['is_legal'].sum():,}")
print(f"Matches: {df['match_id'].nunique():,}")
print(f"Unique batters: {df['striker_id'].nunique():,}")
print(f"Unique bowlers: {df['bowler_id'].nunique():,}")
print(f"Date range: {df['match_date'].min().date()} to {df['match_date'].max().date()}")
print(f"Phase breakdown:")
print(df[df['is_legal']==1]['phase'].value_counts())

# Save as parquet
df.to_parquet(os.path.join(DATA, 'deliveries_clean.parquet'), index=False)
print(f"\nSaved: deliveries_clean.parquet")

# Save Pakistan exclusion
excl_df.to_csv(os.path.join(RES, 'excluded_pakistan_players.csv'), index=False)
print(f"Saved: excluded_pakistan_players.csv ({len(excl_df)} rows)")

# Save scope metadata
scope_meta = {
    'valid_matches': len(valid_mids),
    'ipl_matches': ipl_m,
    't20i_matches': t20i_m,
    'total_deliveries': len(bbb_all),
    'legal_deliveries': int(df['is_legal'].sum()),
    'excluded_pakistan_players': len(all_pak_ids),
    'date_min': str(df['match_date'].min().date()),
    'date_max': str(df['match_date'].max().date()),
    'seed': SEED,
}
with open(os.path.join(DATA, 'scope_meta.json'), 'w') as f:
    json.dump(scope_meta, f, indent=2)

print("\n=== STEP 0 + 1 SUMMARY ===")
print(f"  Valid T20 matches (IPL+T20I, 2022+): {len(valid_mids)}")
print(f"  IPL: {ipl_m}  |  T20I: {t20i_m}")
print(f"  Pakistan players excluded: {len(all_pak_ids)} (matches kept for context)")
print(f"  Clean deliveries: {len(df):,}  |  Legal: {df['is_legal'].sum():,}")
print(f"  bowl_speed available: {spd_ok:,} deliveries ({100*spd_ok/len(df):.1f}%)")
print(f"  delivery_style: {bbb['delivery_style'].nunique()} types (Yorker, Bouncer etc.)")
print(f"  stroke: {bbb['stroke'].nunique()} shot types")
print(f"  over_or_around + on_or_off: available")
print("STEPS 0+1 COMPLETE ✓")
