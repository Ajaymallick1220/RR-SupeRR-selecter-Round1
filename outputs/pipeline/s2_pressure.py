"""
s2_pressure.py – Win-Probability Model + Pressure Definition + Scenario Tagging (Steps 2 & 2B)
SupeRR Selector S2 (v7)
"""
import warnings, json, pathlib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
import joblib, os

SEED = 42
np.random.seed(SEED)

WORK  = pathlib.Path(r'D:\SupeRR_round_1')
CLEAN = WORK / 'outputs' / 'data_clean'
RES   = WORK / 'outputs' / 'results'
FIG   = WORK / 'outputs' / 'figures'
PIPE  = WORK / 'outputs' / 'pipeline'
for d in [RES, FIG, PIPE]: d.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 2 – WIN-PROBABILITY MODEL + PRESSURE DEFINITION")
print("=" * 70)

# ── Load clean data ─────────────────────────────────────────────────────────
print("\nLoading deliveries_clean.parquet …")
df = pd.read_parquet(CLEAN / 'deliveries_clean.parquet')
print(f"  Rows: {len(df):,}   Cols: {df.shape[1]}")

excl_df = pd.read_csv(RES / 'excluded_pakistan_players.csv')
PAK_IDS = set(excl_df['player_id'].tolist())

# Keep only legal, non-super-over rows for WP model
legal = df[(df['is_legal'] == 1) & (df['is_super_over'] == 0)].copy()
print(f"  Legal non-super-over: {len(legal):,}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_top_venue(ser, n=20):
    top = ser.value_counts().head(n).index.tolist()
    return ser.where(ser.isin(top), other='other')

# ── INNINGS-2 WP MODEL (GBM) ─────────────────────────────────────────────────
print("\n--- Innings-2 WP Model (GBM) ---")
inn2 = legal[legal['innings'] == 2].copy()
inn2['rrr'] = inn2['rrr'].fillna(6.0 * inn2['runs_required'])
inn2['rrr_minus_crr'] = inn2['rrr_minus_crr'].fillna(inn2['rrr'] - inn2['crr'])
inn2 = inn2.dropna(subset=['runs_required', 'legal_balls_remaining', 'wickets_in_hand', 'rrr', 'crr', 'over_0based'])

# Target: did batting team win this match?
inn2['bat_won'] = (inn2['batting_team_id'] == inn2['win_team_id']).astype(int)

# Venue top-20
inn2['venue_grp'] = safe_top_venue(inn2['venue_id'].astype(str))

# Phase dummies
phase_dummies = pd.get_dummies(inn2['phase'], prefix='ph')
# Venue dummies
venue_dummies = pd.get_dummies(inn2['venue_grp'], prefix='vn')

FEATS_INN2 = ['runs_required', 'legal_balls_remaining', 'wickets_in_hand',
              'rrr', 'crr', 'over_0based', 'is_ipl', 'batter_is_set']
inn2_X = pd.concat([
    inn2[FEATS_INN2].fillna(0).reset_index(drop=True),
    phase_dummies.reset_index(drop=True),
    venue_dummies.reset_index(drop=True)
], axis=1)
inn2_y = inn2['bat_won'].values
inn2_groups = inn2['match_id'].values

gkf = GroupKFold(n_splits=5)
inn2_oof_proba = np.zeros(len(inn2_y))
inn2_oof_idx   = np.zeros(len(inn2_y), dtype=int)

gbm2 = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=50, random_state=SEED
)

fold_metrics = []
for fold, (tr, te) in enumerate(gkf.split(inn2_X, inn2_y, inn2_groups)):
    gbm2.fit(inn2_X.iloc[tr], inn2_y[tr])
    phat = gbm2.predict_proba(inn2_X.iloc[te])[:, 1]
    inn2_oof_proba[te] = phat
    ll  = log_loss(inn2_y[te], phat)
    br  = brier_score_loss(inn2_y[te], phat)
    fold_metrics.append({'fold': fold+1, 'log_loss': ll, 'brier': br})
    print(f"  Fold {fold+1}: log_loss={ll:.4f}  brier={br:.4f}")

fm = pd.DataFrame(fold_metrics)
print(f"\n  OOF mean log_loss={fm.log_loss.mean():.4f}  brier={fm.brier.mean():.4f}")

# Retrain on all for prediction
gbm2.fit(inn2_X, inn2_y)
joblib.dump(gbm2, PIPE / 'gbm_inn2.pkl')

# Chronological split
chron_cut = '2024-01-01'
# Reset index for inn2_X so masks align properly
inn2_X_reset = inn2_X.reset_index(drop=True)
inn2_y_arr   = inn2_y  # already numpy array
inn2_dates   = inn2['match_date'].astype(str).reset_index(drop=True)
tr_mask_arr  = (inn2_dates < chron_cut).values
te_mask_arr  = ~tr_mask_arr
if tr_mask_arr.sum() > 1000 and te_mask_arr.sum() > 500:
    gbm2_chron = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=50, random_state=SEED
    )
    gbm2_chron.fit(inn2_X_reset[tr_mask_arr], inn2_y_arr[tr_mask_arr])
    phat_chron = gbm2_chron.predict_proba(inn2_X_reset[te_mask_arr])[:, 1]
    chron_ll = log_loss(inn2_y_arr[te_mask_arr], phat_chron)
    chron_br = brier_score_loss(inn2_y_arr[te_mask_arr], phat_chron)
    print(f"  Chronological split (train<2024, test>=2024): log_loss={chron_ll:.4f}  brier={chron_br:.4f}")
else:
    chron_ll, chron_br = None, None
    print("  Chronological split: insufficient data in one partition")

# Calibration curve (OOF only)
fig, ax = plt.subplots(figsize=(6, 5))
frac_pos, mean_pred = calibration_curve(inn2_y, inn2_oof_proba, n_bins=10)
ax.plot(mean_pred, frac_pos, 'b-o', label='GBM Inn-2 (OOF)')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
ax.set_title('Innings-2 WP Model – Calibration Curve (OOF)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / 'calibration_inn2.png', dpi=150)
plt.close(fig)
print("  Saved calibration_inn2.png")

# ── INNINGS-1 WP MODEL (Logistic Regression) ─────────────────────────────────
print("\n--- Innings-1 WP Model (Logistic Regression) ---")
inn1 = legal[legal['innings'] == 1].copy()
inn1 = inn1.dropna(subset=['team_runs_so_far', 'wickets_so_far', 'over_0based', 'venue_par'])

inn1['bat_won'] = (inn1['batting_team_id'] == inn1['win_team_id']).astype(int)
inn1['rpo_vs_par'] = inn1['team_runs_so_far'] / (inn1['over_0based'].clip(lower=1)) - inn1['venue_par'] / 20
inn1['venue_grp'] = safe_top_venue(inn1['venue_id'].astype(str))

ph1_dummies = pd.get_dummies(inn1['phase'], prefix='ph')
vn1_dummies = pd.get_dummies(inn1['venue_grp'], prefix='vn')

FEATS_INN1 = ['team_runs_so_far', 'wickets_so_far', 'over_0based', 'venue_par',
              'rpo_vs_par', 'is_ipl']
inn1_X = pd.concat([
    inn1[FEATS_INN1].fillna(0).reset_index(drop=True),
    ph1_dummies.reset_index(drop=True),
    vn1_dummies.reset_index(drop=True)
], axis=1)
inn1_y     = inn1['bat_won'].values
inn1_groups= inn1['match_id'].values

lr1 = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
inn1_oof_proba = np.zeros(len(inn1_y))

fold_metrics1 = []
for fold, (tr, te) in enumerate(gkf.split(inn1_X, inn1_y, inn1_groups)):
    lr1.fit(inn1_X.iloc[tr], inn1_y[tr])
    phat = lr1.predict_proba(inn1_X.iloc[te])[:, 1]
    inn1_oof_proba[te] = phat
    ll = log_loss(inn1_y[te], phat)
    br = brier_score_loss(inn1_y[te], phat)
    fold_metrics1.append({'fold': fold+1, 'log_loss': ll, 'brier': br})
    print(f"  Fold {fold+1}: log_loss={ll:.4f}  brier={br:.4f}")

fm1 = pd.DataFrame(fold_metrics1)
print(f"\n  OOF mean log_loss={fm1.log_loss.mean():.4f}  brier={fm1.brier.mean():.4f}")
if fm1.brier.mean() > 0.22:
    print("  NOTE: Inn-1 Brier > 0.22 as expected (less predictive); WPA from Inn-1 used with caution")

lr1.fit(inn1_X, inn1_y)
joblib.dump(lr1, PIPE / 'lr_inn1.pkl')

# Calibration curve inn1
fig, ax = plt.subplots(figsize=(6, 5))
frac_pos1, mean_pred1 = calibration_curve(inn1_y, inn1_oof_proba, n_bins=10)
ax.plot(mean_pred1, frac_pos1, 'r-o', label='LR Inn-1 (OOF)')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
ax.set_title('Innings-1 WP Model – Calibration Curve (OOF)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / 'calibration_inn1.png', dpi=150)
plt.close(fig)
print("  Saved calibration_inn1.png")

# ── ATTACH WP BEFORE / AFTER PER BALL ────────────────────────────────────────
print("\n--- Attaching wp_before / wp_after per legal ball ---")

# We'll compute WP on each ball's state BEFORE the delivery, then shift by 1
# For inn2: predict WP before each ball
inn2_wp_before = gbm2.predict_proba(inn2_X)[:, 1]
inn2_df = inn2.copy()
inn2_df['wp_before'] = inn2_wp_before
# wp_after = wp_before of NEXT ball in same match-innings; for last ball use match outcome
inn2_df = inn2_df.sort_values(['match_id', 'innings', 'legal_ball_seq'])
inn2_df['wp_after'] = inn2_df.groupby(['match_id', 'innings'])['wp_before'].shift(-1)
# Fill last ball with match outcome
inn2_df['wp_after'] = inn2_df['wp_after'].fillna(inn2_df['bat_won'].astype(float))

for col in ['bat_won', 'venue_grp']:
    if col in inn2_df.columns:
        inn2_df.drop(columns=[col], inplace=True)
# Drop dummy cols we added
for col in list(phase_dummies.columns) + list(venue_dummies.columns):
    if col in inn2_df.columns:
        inn2_df.drop(columns=[col], inplace=True)

# For inn1
inn1_wp_before = lr1.predict_proba(inn1_X)[:, 1]
inn1_X_after = inn1_X.copy()
inn1_X_after['team_runs_so_far'] = inn1['team_runs_so_far'].values + inn1['total_del_runs'].values
inn1_X_after['wickets_so_far'] = (inn1['wickets_so_far'].values + inn1['is_wicket'].values).clip(max=10)
inn1_X_after['over_0based'] = (inn1['legal_balls_bowled'].values // 6).clip(max=20)
inn1_X_after['rpo_vs_par'] = inn1_X_after['team_runs_so_far'] / inn1_X_after['over_0based'].clip(lower=1) - inn1_X_after['venue_par'] / 20
inn1_wp_end = lr1.predict_proba(inn1_X_after)[:, 1]
inn1_df = inn1.copy()
inn1_df['wp_before'] = inn1_wp_before
inn1_df['wp_end_calc'] = inn1_wp_end
inn1_df = inn1_df.sort_values(['match_id', 'innings', 'legal_ball_seq'])
inn1_df['wp_after'] = inn1_df.groupby(['match_id', 'innings'])['wp_before'].shift(-1)
inn1_df['wp_after'] = inn1_df['wp_after'].fillna(inn1_df['wp_end_calc'])
inn1_df.drop(columns=['wp_end_calc'], inplace=True)

for col in ['bat_won', 'rpo_vs_par', 'venue_grp']:
    if col in inn1_df.columns:
        inn1_df.drop(columns=[col], inplace=True)
for col in list(ph1_dummies.columns) + list(vn1_dummies.columns):
    if col in inn1_df.columns:
        inn1_df.drop(columns=[col], inplace=True)

# WPA from BATTING TEAM perspective: wpa_batter = wp_after - wp_before
inn2_df['wpa_batter'] = inn2_df['wp_after'] - inn2_df['wp_before']
inn2_df['wpa_bowler'] = -inn2_df['wpa_batter']
inn1_df['wpa_batter'] = inn1_df['wp_after'] - inn1_df['wp_before']
inn1_df['wpa_bowler'] = -inn1_df['wpa_batter']

print(f"  Inn-2 WPA: mean={inn2_df.wpa_batter.mean():.4f}  std={inn2_df.wpa_batter.std():.4f}")
print(f"  Inn-1 WPA: mean={inn1_df.wpa_batter.mean():.4f}  std={inn1_df.wpa_batter.std():.4f}")

# Merge back to full legal dataframe
legal_wp = pd.concat([inn1_df, inn2_df], ignore_index=True)
print(f"  Rows with WP attached: {len(legal_wp):,}")

# ── LEVERAGE INDEX ─────────────────────────────────────────────────────────────
print("\n--- Computing Leverage Index ---")
# LI = weighted mean |ΔWP| across 4 outcomes / global mean |ΔWP|
# Empirical outcome rates
boundary_rate = (legal_wp['runs_bat'] >= 4).mean()
wicket_rate   = legal_wp['is_wicket_bat'].mean()
dot_rate      = (legal_wp['runs_bat'] == 0).mean()
single_rate   = 1 - boundary_rate - wicket_rate

global_mean_dwp = legal_wp['wpa_batter'].abs().mean()
print(f"  Global mean |WPA|: {global_mean_dwp:.5f}")

# For LI, we approximate |ΔWP| as a function of match state (abs(wpa_batter) is a proxy)
# Since we can't cheaply simulate per-ball, use |wpa_batter| as LI proxy (normalised)
legal_wp['li'] = (legal_wp['wpa_batter'].abs() / global_mean_dwp).clip(upper=10)
print(f"  LI mean={legal_wp.li.mean():.3f}  median={legal_wp.li.median():.3f}  max(clipped)={legal_wp.li.max():.2f}")

# LI by over (mean LI per over for each innings)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, inn, label in zip(axes, [1, 2], ['Innings 1', 'Innings 2']):
    sub = legal_wp[legal_wp['innings'] == inn]
    li_by_over = sub.groupby('over_0based')['li'].mean()
    ax.bar(li_by_over.index + 1, li_by_over.values, color='steelblue', alpha=0.75)
    ax.set_xlabel('Over'); ax.set_ylabel('Mean Leverage Index')
    ax.set_title(f'Leverage Index by Over – {label}')
    ax.axhline(1.0, color='red', linestyle='--', label='Mean LI=1')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / 'li_by_over.png', dpi=150)
plt.close(fig)
print("  Saved li_by_over.png")

# ── BATTER PRESSURE SCORE + BOWLER PRESSURE SCORE ────────────────────────────
print("\n--- Computing batter and bowler pressure scores ---")
# Batter pressure rises with: rrr_minus_crr, low wickets_in_hand, batter_is_new,
# dots_before, is_knockout, phase=death

def normalize_01(s):
    mn, mx = s.min(), s.max()
    if mx == mn: return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

# Batter-pressure components (all 0-1 normalised, then weighted sum)
batt_p = pd.DataFrame(index=legal_wp.index)
batt_p['rrr_component']  = normalize_01(legal_wp['rrr_minus_crr'].clip(-5, 15))
batt_p['wikets_component'] = normalize_01(10 - legal_wp['wickets_in_hand'].clip(1, 10))  # low WIH = high pressure
batt_p['new_batter']     = legal_wp['batter_is_new'].astype(float)
batt_p['dots_component'] = normalize_01(legal_wp['dots_before'].clip(0, 6))
batt_p['is_knockout']    = legal_wp['is_knockout'].astype(float)
batt_p['death_phase']    = (legal_wp['phase'] == 'death').astype(float)
batt_p['li_component']   = normalize_01(legal_wp['li'])

# Weights
W_BATT = np.array([0.30, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10])
legal_wp['batter_pressure'] = (batt_p.values * W_BATT).sum(axis=1)

# Bowler-pressure components
bowl_p = pd.DataFrame(index=legal_wp.index)
bowl_p['crr_vs_rrr']     = normalize_01(legal_wp['crr'] - legal_wp['rrr_minus_crr'].clip(-5, 15))  # batting team ahead = bowling pressure up
bowl_p['wickets_gone']   = normalize_01(legal_wp['wickets_so_far'].clip(0, 9))
bowl_p['set_batter']     = legal_wp['batter_is_set'].astype(float)
bowl_p['death_phase']    = (legal_wp['phase'] == 'death').astype(float)
bowl_p['is_knockout']    = legal_wp['is_knockout'].astype(float)
bowl_p['dots_build']     = normalize_01(legal_wp['dots_before'].clip(0, 6))
bowl_p['li_component']   = normalize_01(legal_wp['li'])

W_BOWL = np.array([0.25, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10])
legal_wp['bowler_pressure'] = (bowl_p.values * W_BOWL).sum(axis=1)

print(f"  batter_pressure: mean={legal_wp.batter_pressure.mean():.3f}  std={legal_wp.batter_pressure.std():.3f}")
print(f"  bowler_pressure: mean={legal_wp.bowler_pressure.mean():.3f}  std={legal_wp.bowler_pressure.std():.3f}")

# Prove it works: boundary/wicket/dot rates in top-25% pressure vs bottom-25%
q75b = legal_wp['batter_pressure'].quantile(0.75)
q25b = legal_wp['batter_pressure'].quantile(0.25)
hi_p = legal_wp[legal_wp['batter_pressure'] >= q75b]
lo_p = legal_wp[legal_wp['batter_pressure'] <= q25b]
print(f"\n  Batter pressure validation:")
print(f"    HIGH pressure (top-25%): boundary%={100*(hi_p.runs_bat>=4).mean():.1f}  wicket%={100*hi_p.is_wicket_bat.mean():.1f}  dot%={100*(hi_p.runs_bat==0).mean():.1f}")
print(f"    LOW  pressure (bot-25%): boundary%={100*(lo_p.runs_bat>=4).mean():.1f}  wicket%={100*lo_p.is_wicket_bat.mean():.1f}  dot%={100*(lo_p.runs_bat==0).mean():.1f}")

q75w = legal_wp['bowler_pressure'].quantile(0.75)
q25w = legal_wp['bowler_pressure'].quantile(0.25)
hi_w = legal_wp[legal_wp['bowler_pressure'] >= q75w]
lo_w = legal_wp[legal_wp['bowler_pressure'] <= q25w]
print(f"\n  Bowler pressure validation:")
print(f"    HIGH pressure (top-25%): boundary%={100*(hi_w.runs_bat>=4).mean():.1f}  wicket%={100*hi_w.is_wicket_bat.mean():.1f}  dot%={100*(hi_w.runs_bat==0).mean():.1f}")
print(f"    LOW  pressure (bot-25%): boundary%={100*(lo_w.runs_bat>=4).mean():.1f}  wicket%={100*lo_w.is_wicket_bat.mean():.1f}  dot%={100*(lo_w.runs_bat==0).mean():.1f}")

# ── STEP 2B – SCENARIO TAGGING ────────────────────────────────────────────────
print("\n--- Step 2B: Scenario Tagging (A through I) ---")

# Identify collapse windows: 2+ wickets in last 12 balls or 3+ in 24
# Compute wickets in rolling window per match-innings
legal_wp_sorted = legal_wp.sort_values(['match_id', 'innings', 'legal_ball_seq'])
legal_wp_sorted['roll_wkts_12'] = (
    legal_wp_sorted.groupby(['match_id', 'innings'])['is_wicket_bat']
    .transform(lambda x: x.rolling(12, min_periods=1).sum())
)
legal_wp_sorted['roll_wkts_24'] = (
    legal_wp_sorted.groupby(['match_id', 'innings'])['is_wicket_bat']
    .transform(lambda x: x.rolling(24, min_periods=1).sum())
)

wp = legal_wp_sorted  # alias

# Scenario A: Chasing (Innings 2 specific)
scen_A_tight      = (wp['innings'] == 2) & (wp['legal_balls_remaining'] <= 18) & (wp['wp_before'].between(0.25, 0.75))
scen_A_climbing   = (wp['innings'] == 2) & (wp['rrr_minus_crr'] >= 2)
scen_A_collapse   = (wp['innings'] == 2) & ((wp['roll_wkts_12'] >= 2) | (wp['roll_wkts_24'] >= 3))
scen_A_last_over  = (wp['innings'] == 2) & (wp['over_0based'] >= 19)
wp['scen_A'] = (scen_A_tight | scen_A_climbing | scen_A_collapse | scen_A_last_over).astype(int)

# Scenario B: Setting (Innings 1)
scen_B_early_coll = (wp['innings'] == 1) & (wp['over_0based'] <= 5) & (wp['wickets_so_far'] >= 3)
scen_B_rebuild    = (wp['innings'] == 1) & (wp['over_0based'] >= 10) & (wp['wickets_so_far'] >= 4) & (wp['wp_before'] < 0.5)
scen_B_death_push = (wp['innings'] == 1) & (wp['phase'] == 'death')
wp['scen_B'] = (scen_B_early_coll | scen_B_rebuild | scen_B_death_push).astype(int)

# Scenario C: Knockout / final / qualifier
wp['scen_C'] = wp['is_knockout'].astype(int)

# Scenario D: Moment-of-truth
scen_D_new_death  = (wp['batter_is_new'] == 1) & (wp['phase'] == 'death')
scen_D_dots_build = wp['dots_before'] >= 3
scen_D_def_small  = (wp['innings'] == 2) & (wp['over_0based'] >= 17) & (wp['runs_required'] <= 15) & (wp['runs_required'] > 0)
wp['scen_D'] = (scen_D_new_death | scen_D_dots_build | scen_D_def_small).astype(int)

# Scenario E: Big ask grid (balls_remaining × runs_needed buckets)
# Tag balls where rrr >= 10+ or runs_required >= 30 in last 5 overs
scen_E_big_ask = (wp['innings'] == 2) & (wp['rrr'] >= 10) & (wp['over_0based'] >= 15)
wp['scen_E'] = scen_E_big_ask.astype(int)

# Scenario F: Collapse and rebuild (batting side 3+ wkts in first 8 overs → next batters)
scen_F = (wp['wickets_so_far'] >= 3) & (wp['over_0based'] <= 8)
wp['scen_F'] = scen_F.astype(int)

# Scenario G: High-scoring pitch (venue-season RPO > upper quartile)
match_rpo_q75 = wp['match_rpo_per_ball'].quantile(0.75)
wp['scen_G'] = (wp['match_rpo_per_ball'] >= match_rpo_q75).astype(int)

# Scenario H: Pressure transfer / turning points (ΔWP ≥ 15pp in ball)
wp['scen_H'] = (wp['wpa_batter'].abs() >= 0.15).astype(int)

# Scenario I: Spells / bounce-back / resilience
# Heavy overs: a bowler concedes >= 15 runs in an over → flag those overs and next
# Compute runs per over per bowler
over_runs = (
    wp.groupby(['match_id', 'innings', 'bowler_id', 'over_0based'])['total_del_runs']
    .sum().reset_index().rename(columns={'total_del_runs': 'over_runs_conceded'})
)
over_runs['heavy_over'] = (over_runs['over_runs_conceded'] >= 15).astype(int)
wp = wp.merge(over_runs[['match_id', 'innings', 'bowler_id', 'over_0based', 'heavy_over',
                           'over_runs_conceded']],
              on=['match_id', 'innings', 'bowler_id', 'over_0based'], how='left')
wp['heavy_over'] = wp['heavy_over'].fillna(0).astype(int)

# Bounce-back: ball in an over immediately after a heavy over
# Tag next over for the same bowler in same match-innings
# Compute previous over runs (shifted)
over_runs_sorted = over_runs.sort_values(['match_id', 'innings', 'bowler_id', 'over_0based'])
over_runs_sorted['prev_over_runs'] = over_runs_sorted.groupby(['match_id', 'innings', 'bowler_id'])['over_runs_conceded'].shift(1)
over_runs_sorted['prev_heavy']     = (over_runs_sorted['prev_over_runs'] >= 15).fillna(False).astype(int)
wp = wp.merge(over_runs_sorted[['match_id', 'innings', 'bowler_id', 'over_0based', 'prev_heavy']],
              on=['match_id', 'innings', 'bowler_id', 'over_0based'], how='left')
wp['prev_heavy'] = wp['prev_heavy'].fillna(0).astype(int)

# Return vs set batter (batter_is_set = True, batter facing bowler again)
wp['scen_I_set_batter'] = (wp['batter_is_set'] == 1).astype(int)
wp['scen_I'] = ((wp['prev_heavy'] == 1) | (wp['scen_I_set_batter'] == 1)).astype(int)

# Overall pressure scenario label (which scenario(s) apply)
scen_cols = ['scen_A', 'scen_B', 'scen_C', 'scen_D', 'scen_E', 'scen_F', 'scen_G', 'scen_H', 'scen_I']
wp['n_scenarios'] = wp[scen_cols].sum(axis=1)
wp['is_pressure_ball'] = (wp['n_scenarios'] >= 1).astype(int)

print("\n  Scenario tag counts (balls):")
for s in scen_cols:
    print(f"    {s}: {wp[s].sum():,}  ({100*wp[s].mean():.1f}%)")
print(f"    Any scenario: {wp.is_pressure_ball.sum():,}  ({100*wp.is_pressure_ball.mean():.1f}%)")

# ── PRESSURE vs ORDINARY by PHASE ─────────────────────────────────────────────
print("\n--- Pressure vs Ordinary by Phase ---")
for phase in ['powerplay', 'middle', 'death']:
    sub = wp[wp['phase'] == phase]
    p_sub = sub[sub['is_pressure_ball'] == 1]
    o_sub = sub[sub['is_pressure_ball'] == 0]
    if len(p_sub) < 100 or len(o_sub) < 100:
        continue
    print(f"\n  Phase: {phase.upper()}")
    print(f"    PRESSURE  — boundary%={100*(p_sub.runs_bat>=4).mean():.1f}  wicket%={100*p_sub.is_wicket_bat.mean():.1f}  dot%={100*(p_sub.runs_bat==0).mean():.1f}  n={len(p_sub):,}")
    print(f"    ORDINARY  — boundary%={100*(o_sub.runs_bat>=4).mean():.1f}  wicket%={100*o_sub.is_wicket_bat.mean():.1f}  dot%={100*(o_sub.runs_bat==0).mean():.1f}  n={len(o_sub):,}")

# Figure: pressure vs ordinary
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_labels = ['Boundary %', 'Wicket %', 'Dot %']
phases = ['powerplay', 'middle', 'death']
colors = ['steelblue', 'darkorange']
for ax, phase in zip(axes, phases):
    sub  = wp[wp['phase'] == phase]
    p_sub= sub[sub['is_pressure_ball'] == 1]
    o_sub= sub[sub['is_pressure_ball'] == 0]
    vals = [
        [100*(p_sub.runs_bat>=4).mean(), 100*(o_sub.runs_bat>=4).mean()],
        [100*p_sub.is_wicket_bat.mean(),  100*o_sub.is_wicket_bat.mean()],
        [100*(p_sub.runs_bat==0).mean(),  100*(o_sub.runs_bat==0).mean()],
    ]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, [v[0] for v in vals], w, color=colors[0], label='Pressure')
    ax.bar(x + w/2, [v[1] for v in vals], w, color=colors[1], label='Ordinary')
    ax.set_xticks(x); ax.set_xticklabels(metrics_labels, fontsize=9)
    ax.set_title(f'{phase.capitalize()}')
    ax.set_ylabel('%'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
fig.suptitle('Pressure vs Ordinary Deliveries by Phase', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'pressure_vs_ordinary_phase.png', dpi=150)
plt.close(fig)
print("\n  Saved pressure_vs_ordinary_phase.png")

# ── SAVE deliveries_pressure.parquet ─────────────────────────────────────────
print("\n--- Saving deliveries_pressure.parquet ---")
# Drop helper columns from over_runs merge that may be duplicated
cols_to_drop = [c for c in ['over_runs_conceded', 'roll_wkts_12', 'roll_wkts_24',
                              'prev_over_runs', 'scen_I_set_batter'] if c in wp.columns]
wp.drop(columns=cols_to_drop, inplace=True, errors='ignore')

wp.to_parquet(CLEAN / 'deliveries_pressure.parquet', index=False)
print(f"  Saved: {CLEAN / 'deliveries_pressure.parquet'}")
print(f"  Shape: {wp.shape}")
print(f"  Columns: {list(wp.columns)}")

# Save model metrics
metrics_dict = {
    'inn2_gbm': {
        'oof_log_loss': round(fm.log_loss.mean(), 4),
        'oof_brier':    round(fm.brier.mean(), 4),
        'chron_log_loss': round(chron_ll, 4) if chron_ll else None,
        'chron_brier':    round(chron_br, 4) if chron_br else None,
    },
    'inn1_lr': {
        'oof_log_loss': round(fm1.log_loss.mean(), 4),
        'oof_brier':    round(fm1.brier.mean(), 4),
    }
}
with open(RES / 'wp_model_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print("\n  Saved wp_model_metrics.json")

print("\n" + "=" * 70)
print("STEP 2 + 2B COMPLETE")
print(f"  deliveries_pressure.parquet: {len(wp):,} rows")
print(f"  Pressure balls: {wp.is_pressure_ball.sum():,} ({100*wp.is_pressure_ball.mean():.1f}%)")
print("=" * 70)
