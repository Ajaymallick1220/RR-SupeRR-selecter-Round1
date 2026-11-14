"""
s6_validation.py – Out-of-time validation, split-half, robustness checks (Step 6)
SupeRR Selector S2 (v7)
"""
import warnings, pathlib, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

SEED = 42
np.random.seed(SEED)
rng  = np.random.default_rng(SEED)

WORK  = pathlib.Path(r'D:\SupeRR_round_1')
CLEAN = WORK / 'outputs' / 'data_clean'
RES   = WORK / 'outputs' / 'results'
FIG   = WORK / 'outputs' / 'figures'

print("=" * 70)
print("STEP 6 – VALIDATION")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
bat_mi   = pd.read_parquet(CLEAN / 'bat_match_innings_adj.parquet')
bowl_mi  = pd.read_parquet(CLEAN / 'bowl_match_innings_adj.parquet')
bat_elig  = pd.read_csv(RES / 'bat_eligible_ranked.csv')
bowl_elig = pd.read_csv(RES / 'bowl_eligible_ranked.csv')
excl_df   = pd.read_csv(RES / 'excluded_pakistan_players.csv')
PAK_IDS   = set(excl_df['player_id'].tolist())

ctpm = pd.read_csv(r'D:\SupeRR_round_1\competition_team_player_mapping.csv')
name_map = dict(zip(ctpm['player_id'], ctpm['player_name']))

LN2 = np.log(2)
REF_DATE = pd.Timestamp('2026-11-01')
bat_mi['match_date']  = pd.to_datetime(bat_mi['match_date'])
bowl_mi['match_date'] = pd.to_datetime(bowl_mi['match_date'])

def add_recency_weight(mi_df, date_col='match_date', half_life_months=12):
    hl_days = half_life_months * 30.44
    lam = LN2 / hl_days
    age_days = (REF_DATE - mi_df[date_col]).dt.days.clip(lower=0).astype(float)
    return np.exp(-lam * age_days)

shrinkage = json.load(open(RES / 'shrinkage_constants.json'))
C_bat  = shrinkage['C_bat']
C_bowl = shrinkage['C_bowl']

def compute_wpae_weighted(mi_df, pid_col, wpae_col, C, hl=12):
    mi_df = mi_df.copy()
    mi_df['w'] = add_recency_weight(mi_df, half_life_months=hl)
    agg = mi_df.groupby(pid_col).apply(
        lambda g: pd.Series({
            'wpae_raw': (g[wpae_col] * g['w']).sum() / max(g['w'].sum(), 1e-9),
            'n':         len(g),
        })
    ).reset_index()
    agg['wpae_shrunk'] = (agg['n'] / (agg['n'] + C)) * agg['wpae_raw']
    return agg.set_index(pid_col)['wpae_shrunk']

print(f"  Bat MI: {len(bat_mi):,}   Bowl MI: {len(bowl_mi):,}")

# ── 1. OUT-OF-TIME VALIDATION ──────────────────────────────────────────────────
print("\n--- 1. Out-of-Time Validation ---")

oot_results = []
for train_end, test_start, label in [
    ('2024-01-01', '2024-01-01', 'Train 2022-23 → Test 2024-26'),
    ('2025-01-01', '2025-01-01', 'Train 2022-24 → Test 2025-26'),
]:
    te = pd.Timestamp(test_start)
    
    bat_tr = bat_mi[bat_mi['match_date'] < te]
    bat_te = bat_mi[bat_mi['match_date'] >= te]
    
    if len(bat_tr) < 500 or len(bat_te) < 200:
        print(f"  [{label}]: insufficient data")
        continue
    
    # Players appearing in both train and test
    tr_pids = set(bat_tr['striker_id'].unique())
    te_pids = set(bat_te['striker_id'].unique())
    common_pids = tr_pids & te_pids - PAK_IDS
    
    if len(common_pids) < 20:
        print(f"  [{label}]: only {len(common_pids)} common players")
        continue
    
    # Train ratings
    tr_ratings = compute_wpae_weighted(bat_tr[bat_tr['striker_id'].isin(common_pids)],
                                       'striker_id', 'wpae_bat_adj', C_bat)
    # Test performance (plain sum, no decay for test)
    te_perf = bat_te[bat_te['striker_id'].isin(common_pids)].groupby('striker_id')['wpae_bat_adj'].mean()
    
    merged = pd.DataFrame({'train': tr_ratings, 'test': te_perf}).dropna()
    if len(merged) < 15:
        print(f"  [{label}]: only {len(merged)} players after merge")
        continue
    
    rho, pval = spearmanr(merged['train'], merged['test'])
    oot_results.append({'split': label, 'n_players': len(merged), 'spearman_rho': round(rho,3), 'p_value': round(pval,4)})
    print(f"  [{label}]: n={len(merged)} players, Spearman rho={rho:.3f} (p={pval:.4f})")

# Same for bowlers
for train_end, test_start, label in [
    ('2024-01-01', '2024-01-01', 'Bowl Train 2022-23 → Test 2024-26'),
    ('2025-01-01', '2025-01-01', 'Bowl Train 2022-24 → Test 2025-26'),
]:
    te = pd.Timestamp(test_start)
    bowl_tr = bowl_mi[bowl_mi['match_date'] < te]
    bowl_te = bowl_mi[bowl_mi['match_date'] >= te]
    if len(bowl_tr) < 300 or len(bowl_te) < 100:
        continue
    tr_pids = set(bowl_tr['bowler_id'].unique())
    te_pids = set(bowl_te['bowler_id'].unique())
    common_pids = tr_pids & te_pids - PAK_IDS
    if len(common_pids) < 15:
        continue
    tr_ratings = compute_wpae_weighted(bowl_tr[bowl_tr['bowler_id'].isin(common_pids)],
                                       'bowler_id', 'wpae_bowl_adj', C_bowl)
    te_perf = bowl_te[bowl_te['bowler_id'].isin(common_pids)].groupby('bowler_id')['wpae_bowl_adj'].mean()
    merged  = pd.DataFrame({'train': tr_ratings, 'test': te_perf}).dropna()
    if len(merged) < 10:
        continue
    rho, pval = spearmanr(merged['train'], merged['test'])
    oot_results.append({'split': label, 'n_players': len(merged), 'spearman_rho': round(rho,3), 'p_value': round(pval,4)})
    print(f"  [{label}]: n={len(merged)} players, Spearman rho={rho:.3f} (p={pval:.4f})")

oot_df = pd.DataFrame(oot_results)
oot_df.to_csv(RES / 'oot_validation.csv', index=False)
print(f"\n  Out-of-time note: Spearman rho in T20 player performance is typically 0.2–0.5; ")
print(f"  T20 is a high-variance format where luck affects individual matches.")
print(f"  We rely on consistent rank above noise floor, not perfect prediction.")

# ── 2. SPLIT-HALF SPEARMAN ───────────────────────────────────────────────────
print("\n--- 2. Split-Half Reliability ---")

N_SPLITS = 100
sh_rhos_bat  = []
sh_rhos_bowl = []

all_matches_bat  = bat_mi['match_id'].unique()
all_matches_bowl = bowl_mi['match_id'].unique()

for _ in range(N_SPLITS):
    # Random half split by match
    half1_m = set(rng.choice(all_matches_bat, size=len(all_matches_bat)//2, replace=False))
    half2_m = set(all_matches_bat) - half1_m
    
    h1 = bat_mi[bat_mi['match_id'].isin(half1_m)]
    h2 = bat_mi[bat_mi['match_id'].isin(half2_m)]
    
    r1 = compute_wpae_weighted(h1, 'striker_id', 'wpae_bat_adj', C_bat)
    r2 = compute_wpae_weighted(h2, 'striker_id', 'wpae_bat_adj', C_bat)
    merged = pd.DataFrame({'h1': r1, 'h2': r2}).dropna()
    if len(merged) >= 20:
        rho, _ = spearmanr(merged['h1'], merged['h2'])
        sh_rhos_bat.append(rho)

sh_bat_mean = np.mean(sh_rhos_bat)
sh_bat_p5   = np.percentile(sh_rhos_bat, 5)
print(f"  Bat split-half Spearman: mean={sh_bat_mean:.3f}  5th_pct={sh_bat_p5:.3f}  (n={len(sh_rhos_bat)} splits)")
if sh_bat_mean < 0.3:
    print("  NOTE: Split-half rho < 0.3 — T20 is a high-variance format; we report this honestly.")
    print("  Bootstrap lower bound (20th pct rank) is used as conservative rank for finalists.")

for _ in range(N_SPLITS):
    half1_m = set(rng.choice(all_matches_bowl, size=len(all_matches_bowl)//2, replace=False))
    half2_m = set(all_matches_bowl) - half1_m
    h1 = bowl_mi[bowl_mi['match_id'].isin(half1_m)]
    h2 = bowl_mi[bowl_mi['match_id'].isin(half2_m)]
    r1 = compute_wpae_weighted(h1, 'bowler_id', 'wpae_bowl_adj', C_bowl)
    r2 = compute_wpae_weighted(h2, 'bowler_id', 'wpae_bowl_adj', C_bowl)
    merged = pd.DataFrame({'h1': r1, 'h2': r2}).dropna()
    if len(merged) >= 20:
        rho, _ = spearmanr(merged['h1'], merged['h2'])
        sh_rhos_bowl.append(rho)

sh_bowl_mean = np.mean(sh_rhos_bowl)
sh_bowl_p5   = np.percentile(sh_rhos_bowl, 5)
print(f"  Bowl split-half Spearman: mean={sh_bowl_mean:.3f}  5th_pct={sh_bowl_p5:.3f}  (n={len(sh_rhos_bowl)} splits)")

# Split-half histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, rhos, label, color in [
    (axes[0], sh_rhos_bat,  'Batters',  'steelblue'),
    (axes[1], sh_rhos_bowl, 'Bowlers',  'darkorange'),
]:
    ax.hist(rhos, bins=20, color=color, alpha=0.75, edgecolor='white')
    ax.axvline(np.mean(rhos), color='red', linestyle='--', label=f'Mean={np.mean(rhos):.2f}')
    ax.axvline(0.3, color='black', linestyle=':', label='rho=0.3')
    ax.set_xlabel('Split-half Spearman ρ'); ax.set_ylabel('Count')
    ax.set_title(f'Split-half Reliability – {label}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('Split-Half Reliability (100 random splits by match)', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG / 'split_half_histogram.png', dpi=150)
plt.close(fig)
print("  Saved split_half_histogram.png")

# ── 3. CRICKET SANITY CHECKS ──────────────────────────────────────────────────
print("\n--- 3. Cricket Sanity Checks ---")

sanity_issues = []
top25 = pd.read_csv(RES / 'top25_unified.csv')
for _, r in top25.iterrows():
    pid  = r['player_id']
    name = r['player_name']
    # Check 1: not tiny sample
    n = r['n_innings_or_spells']
    if n < 10:
        sanity_issues.append({'player_id': pid, 'name': name, 'issue': f'Only {n} innings/spells (small sample)'})
    # Check 2: not one season only
    n_m = r.get('n_matches', 0)
    if n_m < 5:
        sanity_issues.append({'player_id': pid, 'name': name, 'issue': f'Only {n_m} matches'})
    # Check 3: not one competition
    n_c = r.get('n_scenarios_active', 0)
    if n_c == 0:
        sanity_issues.append({'player_id': pid, 'name': name, 'issue': 'No scenario coverage'})
    # Check 4: Pakistan
    if pid in PAK_IDS:
        sanity_issues.append({'player_id': pid, 'name': name, 'issue': 'PAK PLAYER IN TABLE!'})

if sanity_issues:
    print(f"  SANITY ISSUES FOUND ({len(sanity_issues)}):")
    for iss in sanity_issues:
        print(f"    {iss['name']}: {iss['issue']}")
else:
    print("  ✅ All sanity checks passed")

pd.DataFrame(sanity_issues if sanity_issues else [{'result': 'All checks passed'}]).to_csv(
    RES / 'sanity_checks.csv', index=False)

# ── 4. LIMITATIONS SECTION (saved to text) ───────────────────────────────────
limitations = """
LIMITATIONS
===========
1. Data completeness: ball-by-ball data covers IPL 2022–2026 and T20I 2022–2026 (807 matches).
   Bilateral T20I series with limited coverage may underrepresent some players.
2. Win-probability model: the innings-2 GBM achieves Brier ~0.14 (OOF), which is good for
   T20 prediction; the innings-1 LR model is weaker (~0.23) and WPA from innings-1 is
   used with appropriate caution in the WPAE calculation.
3. Split-half reliability: T20 is a high-variance format; individual match outcomes
   contain substantial luck. Measured reliability is LOW: split-half Spearman is about 0.00 (batters) and 0.08 (bowlers); out-of-time Spearman is 0.03-0.18. The rankings describe past pressure performance and are only weakly predictive. Recency weighting, shrinkage and bootstrap CIs are used but cannot remove this noise; read the P(top-10) stability flags as the main uncertainty guide.
4. bowl_speed coverage: 63.9% of balls have valid speed data; mean speed analysis
   is directionally reliable but not exhaustive.
5. Spell inference: bowling spells are inferred from consecutive overs (<=2 overs apart)
   as the data lacks a bowling-end identifier.
6. Elo ratings: computed from home/away team IDs using team-level outcomes, not
   individual player opposition quality; serves as a first-order adjustment only.
7. Impact player rule (IPL 2023+): only balls actually faced/bowled by impact players
   are credited per BCCI rules; there may be edge cases in the data.
8. Pakistan exclusion: 68 player_ids excluded from rankings. Their matches and
   deliveries are retained for context models, opposition Elo, and WP model training.
"""
with open(RES / 'limitations.txt', 'w') as f:
    f.write(limitations)

print("\n  Saved limitations.txt")

# ── SAVE VALIDATION SUMMARY ───────────────────────────────────────────────────
val_summary = {
    'oot_validation': oot_df.to_dict(orient='records') if len(oot_df) > 0 else [],
    'split_half_bat':  {'mean_rho': round(sh_bat_mean,3), 'p5_rho': round(sh_bat_p5,3)},
    'split_half_bowl': {'mean_rho': round(sh_bowl_mean,3),'p5_rho': round(sh_bowl_p5,3)},
    'sanity_issues': len(sanity_issues),
    'pak_in_tables': any(i['issue'].startswith('PAK') for i in sanity_issues),
}
with open(RES / 'validation_summary.json', 'w') as f:
    json.dump(val_summary, f, indent=2)

print("\n" + "=" * 70)
print("STEP 6 COMPLETE")
print("=" * 70)
