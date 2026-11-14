"""
s5_shrinkage.py (v2) - per-appearance recency-weighted WPAE, empirical-Bayes shrinkage,
true match-level bootstrap on the UNIFIED ranking, sensitivity, exclusion-effect.
SupeRR Selector S2 (v7). Seed 42.
"""
import warnings, pathlib, json
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

SEED = 42
rng = np.random.default_rng(SEED)
WORK = pathlib.Path(r'D:\SupeRR_round_1')
CLEAN = WORK/'outputs'/'data_clean'; RES = WORK/'outputs'/'results'; FIG = WORK/'outputs'/'figures'
N_BOOT = 300
REF = pd.Timestamp('2026-11-01'); LN2 = np.log(2)

print("="*70); print("STEP 5 (v2) - SHRINKAGE, BOOTSTRAP, UNIFIED RANKING"); print("="*70)
bat_mi = pd.read_parquet(CLEAN/'bat_match_innings.parquet')
bowl_mi = pd.read_parquet(CLEAN/'bowl_match_innings.parquet')
bat_old = pd.read_csv(RES/'bat_eligible.csv'); bowl_old = pd.read_csv(RES/'bowl_eligible.csv')
legal = pd.read_parquet(CLEAN/'deliveries_wpae.parquet')
PAK = set(pd.read_csv(RES/'excluded_pakistan_players.csv')['player_id'])
ctpm = pd.read_csv(WORK/'competition_team_player_mapping.csv')
name_map = dict(zip(ctpm['player_id'], ctpm['player_name']))
for d in (bat_mi, bowl_mi): d['match_date'] = pd.to_datetime(d['match_date'])

def weights(mi, hl_months):
    if hl_months is None: return np.ones(len(mi))
    lam = LN2/(hl_months*30.44)
    return np.exp(-lam*(REF-mi['match_date']).dt.days.clip(lower=0).astype(float).values)

def est_C(mi, pid, col):
    w = weights(mi, 12)
    m = mi.groupby(pid)[col].mean(); v = mi.groupby(pid)[col].var(); n = mi.groupby(pid)[col].count()
    ws = pd.Series(w, index=mi.index).groupby(mi[pid]).sum()
    ok = n > 1
    noise = (v[ok]*(n[ok]-1)).sum()/(n[ok]-1).sum()
    true = max(m.var() - noise/n.mean(), 1e-9)
    # Scale C to effective recency-weighted matches (ws) with 1-season prior floor
    C_eff = max(noise/true, float(ws[n>=12].median()))
    return C_eff, noise, true
# ---------- opposition strength: separate Elo pools (IPL franchises / national T20I teams) ----------
matches = pd.read_csv(WORK/'matches.csv'); comps = pd.read_csv(WORK/'competitions.csv')
matches = matches.merge(comps[['comp_id','match_format','comp_name']], on='comp_id', how='left')
matches['pool'] = np.where(matches['comp_name'].str.startswith('IPL', na=False), 'ipl', np.where(matches['match_format']=='T20I','t20i','other'))
matches['match_date'] = pd.to_datetime(matches['match_date'])
matches = matches[matches['pool'].isin(['ipl','t20i']) & matches['win_team_id'].notna()].sort_values(['match_date','match_id'])
elo_pre = {}; cur = {}
for pool in ('ipl','t20i'):
    cur = {}
    for _, r in matches[matches['pool']==pool].iterrows():
        a, b = r['home_team_id'], r['away_team_id']; ea, eb = cur.get(a,1500.0), cur.get(b,1500.0)
        elo_pre[(r['match_id'],a)] = ea; elo_pre[(r['match_id'],b)] = eb
        exp_a = 1/(1+10**((eb-ea)/400)); wa = 1.0 if r['win_team_id']==a else 0.0
        cur[a] = ea+20*(wa-exp_a); cur[b] = eb-20*(wa-exp_a)
pool_of = matches.set_index('match_id')['pool'].to_dict()
ev = pd.DataFrame([(k[0],k[1],v) for k,v in elo_pre.items()], columns=['match_id','team_id','elo'])
ev['pool'] = ev['match_id'].map(pool_of); ev['z'] = ev.groupby('pool')['elo'].transform(lambda s: (s-s.mean())/s.std())
zmap = {(r.match_id, r.team_id): r.z for r in ev.itertuples()}
inn_teams = legal.groupby(['match_id','innings'])[['batting_team_id','bowling_team_id']].first().reset_index()
bat_mi = bat_mi.merge(inn_teams, on=['match_id','innings'], how='left'); bowl_mi = bowl_mi.merge(inn_teams, on=['match_id','innings'], how='left')
bat_mi['opp_z'] = [zmap.get((m,t),0.0) for m,t in zip(bat_mi['match_id'], bat_mi['bowling_team_id'])]
bowl_mi['opp_z'] = [zmap.get((m,t),0.0) for m,t in zip(bowl_mi['match_id'], bowl_mi['batting_team_id'])]
bat_mi['pool'] = bat_mi['match_id'].map(pool_of); bowl_mi['pool'] = bowl_mi['match_id'].map(pool_of)
def opp_adjust(mi, col, newcol):
    mi[newcol] = mi[col]
    for pool, g in mi.groupby('pool'):
        beta = np.polyfit(g['opp_z'], g[col], 1)[0]
        mi.loc[g.index, newcol] = g[col] - beta*g['opp_z']
        print(f"  opposition slope [{newcol}, pool={pool}]: {beta:+.4f} per 1 SD of opponent Elo")
opp_adjust(bat_mi, 'wpae_bat_li_sum_capped', 'wpae_bat_adj'); opp_adjust(bowl_mi, 'wpae_bowl_li_sum_capped', 'wpae_bowl_adj')

C_bat, nb, tb = est_C(bat_mi, 'striker_id', 'wpae_bat_adj')
C_bowl, nw, tw = est_C(bowl_mi, 'bowler_id', 'wpae_bowl_adj')
print(f"  Shrinkage C (method of moments): bat={C_bat:.2f} bowl={C_bowl:.2f}")

def rating(mi, pid, val, balls, hl=12, C=5.0):
    w = weights(mi, hl)
    g = pd.DataFrame({'pid': mi[pid].values, 'v': mi[val].values*w, 'w': w, 'b': mi[balls].values*w})
    a = g.groupby('pid').agg(vw=('v','sum'), ws=('w','sum'), bw=('b','sum'), n=('w','size'))
    a['mean'] = a['vw']/a['ws'].clip(lower=1e-9)
    a['shrunk'] = a['vw']/(a['ws']+C)
    return a
THR = {'loose': ((8,50),(8,50)), 'medium': ((15,100),(12,80)), 'strict': ((25,200),(20,150))}

ASSOC_TEAMS = {24, 26, 89, 272, 282, 283, 313, 315, 3886}
bat_teams_map = legal.groupby('striker_id')['batting_team_id'].apply(lambda s: set(s.unique())).to_dict()
bowl_teams_map = legal.groupby('bowler_id')['bowling_team_id'].apply(lambda s: set(s.unique())).to_dict()
ASSOC_ONLY_BAT = {p for p, ts in bat_teams_map.items() if ts and ts.issubset(ASSOC_TEAMS)}
ASSOC_ONLY_BOWL = {p for p, ts in bowl_teams_map.items() if ts and ts.issubset(ASSOC_TEAMS)}
if 'true_sit_pressure' not in legal.columns:
    legal['true_sit_pressure'] = (legal[['scen_A','scen_B','scen_C','scen_D','scen_E','scen_F','scen_H']].sum(axis=1) >= 1).astype(int)
bat_p_balls = legal.groupby('striker_id')['true_sit_pressure'].sum()
bat_p_phase_max = legal[legal['true_sit_pressure']==1].groupby(['striker_id','phase']).size().unstack(fill_value=0).max(axis=1)
bowl_p_balls = legal.groupby('bowler_id')['true_sit_pressure'].sum()
def eligible(a, thr, excl=True, is_bat=True):
    m = (a['n']>=thr[0]) & (a['bw']>=thr[1])
    if is_bat:
        m &= ~a.index.isin(ASSOC_ONLY_BAT)
        m &= (a.index.map(bat_p_balls).fillna(0) >= 85) & (a.index.map(bat_p_phase_max).fillna(0) >= 85)
    else:
        m &= ~a.index.isin(ASSOC_ONLY_BOWL)
        m &= (a.index.map(bowl_p_balls).fillna(0) >= 80)
    if excl: m &= ~a.index.isin(PAK)
    return a[m]

BV, BB = 'wpae_bat_adj', 'balls_faced'
WV, WB = 'wpae_bowl_adj', 'balls_bowled'
bat_all = rating(bat_mi,'striker_id',BV,BB,12,C_bat); bowl_all = rating(bowl_mi,'bowler_id',WV,WB,12,C_bowl)
bat_e = eligible(bat_all, THR['medium'][0], is_bat=True); bowl_e = eligible(bowl_all, THR['medium'][1], is_bat=False)
assert set(bat_e.index)==set(bat_old['player_id']) and set(bowl_e.index)==set(bowl_old['player_id'])
print(f"  Eligible (medium): batters={len(bat_e)} bowlers={len(bowl_e)}")
for k,(bt,wt) in THR.items():
    print(f"  threshold {k}: batters={len(eligible(bat_all,bt,is_bat=True))} bowlers={len(eligible(bowl_all,wt,is_bat=False))}")

# scenario breadth: scenarios (A-I) with >=30 flagged balls and positive summed WPAE
scen = [c for c in legal.columns if c.startswith('scen_')]
def breadth(pidcol, valcol):
    out = {}
    for s in scen:
        f = legal[legal[s]==1]
        g = f.groupby(pidcol)[valcol].agg(['sum','count'])
        out[s] = ((g['sum']>0)&(g['count']>=30))
    return pd.DataFrame(out).fillna(False).sum(axis=1)
bat_breadth = breadth('striker_id','wpae_bat_li'); bowl_breadth = breadth('bowler_id','wpae_bowl_li')

def assemble(e, old, shr, C, breadth_s, valcol):
    d = old.set_index('player_id').join(e[['mean','shrunk','ws']].rename(columns={'mean':f'{valcol}_mean','shrunk':f'{valcol}_shrunk','ws':'n_eff'}))
    d['shrink_factor'] = d['n_eff']/(d['n_eff']+C)
    d['n_scenarios_active'] = breadth_s.reindex(d.index).fillna(0).astype(int)
    return d.reset_index()
bat_elig = assemble(bat_e, bat_old, None, C_bat, bat_breadth, 'wpae_bat')
bowl_elig = assemble(bowl_e, bowl_old, None, C_bowl, bowl_breadth, 'wpae_bowl')
bat_elig['recent_weak'] = (bat_elig['wpae_last12']<0)&(bat_elig['wpae_bat_mean']>0)
bowl_elig['recent_weak'] = (bowl_elig['wpae_last12']<0)&(bowl_elig['wpae_bowl_mean']>0)

# ---------- unified point estimates ----------
def unify(bat_s, bowl_s):
    ids = sorted(set(bat_s.index)|set(bowl_s.index))
    u = pd.DataFrame(index=ids)
    u['bat_wpae'] = bat_s.reindex(ids).fillna(0.0); u['bowl_wpae'] = bowl_s.reindex(ids).fillna(0.0)
    u['wpae_combined'] = u['bat_wpae']+u['bowl_wpae']
    inb = u.index.isin(bat_s.index); inw = u.index.isin(bowl_s.index)
    u['role'] = np.where(inb&inw,'allrounder',np.where(inb,'bat','bowl'))
    return u
U = unify(bat_e['shrunk'], bowl_e['shrunk'])

# ---------- true match-level bootstrap on the unified ranking ----------
print(f"\n--- Bootstrap ({N_BOOT} iterations, match-level resampling with replacement) ---")
all_m = np.unique(np.concatenate([bat_mi['match_id'].unique(), bowl_mi['match_id'].unique()]))
mcode = {m:i for i,m in enumerate(all_m)}; M = len(all_m)
def prep(mi, pid, val, ids, hl=12):
    sub = mi[mi[pid].isin(ids)]
    idx = {p:i for i,p in enumerate(ids)}
    return (sub[pid].map(idx).values, sub['match_id'].map(mcode).values, sub[val].values, weights(sub, hl))
bat_ids = list(bat_e.index); bowl_ids = list(bowl_e.index)
bP = prep(bat_mi,'striker_id',BV,bat_ids); wP = prep(bowl_mi,'bowler_id',WV,bowl_ids)
bn = bat_e['ws'].values.astype(float); wn = bowl_e['ws'].values.astype(float)
uid = list(U.index); ub = {p:i for i,p in enumerate(bat_ids)}; uw = {p:i for i,p in enumerate(bowl_ids)}
bi_u = np.array([ub.get(p,-1) for p in uid]); wi_u = np.array([uw.get(p,-1) for p in uid])
def vec(P, n, C, counts, size):
    code, mc, x, w = P; cw = counts[mc]*w
    num = np.bincount(code, cw*x, minlength=size); den = np.bincount(code, cw, minlength=size)
    return np.where(den>0, num/(den+C), 0.0)
R_u = np.zeros((N_BOOT,len(uid))); R_b = np.zeros((N_BOOT,len(bat_ids))); R_w = np.zeros((N_BOOT,len(bowl_ids)))
S_u = np.zeros_like(R_u); S_b = np.zeros_like(R_b); S_w = np.zeros_like(R_w)
for k in range(N_BOOT):
    counts = rng.multinomial(M, np.full(M,1.0/M))
    sb = vec(bP,bn,C_bat,counts,len(bat_ids)); sw = vec(wP,wn,C_bowl,counts,len(bowl_ids))
    su = np.where(bi_u>=0, sb[np.maximum(bi_u,0)], 0.0)+np.where(wi_u>=0, sw[np.maximum(wi_u,0)], 0.0)
    S_u[k]=su; S_b[k]=sb; S_w[k]=sw
    R_u[k] = (-su).argsort().argsort()+1; R_b[k] = (-sb).argsort().argsort()+1; R_w[k] = (-sw).argsort().argsort()+1
def summ(R,S):
    return pd.DataFrame({'p_top10':(R<=10).mean(0),'p_top25':(R<=25).mean(0),'rank_median':np.median(R,0),
        'rank_p20':np.percentile(R,20,0),'rank_p80':np.percentile(R,80,0),
        'boot_lo':np.percentile(S,2.5,0),'boot_hi':np.percentile(S,97.5,0),'score_p20':np.percentile(S,20,0)})
su_df = summ(R_u,S_u); su_df.index = uid
sb_df = summ(R_b,S_b); sb_df.index = bat_ids
sw_df = summ(R_w,S_w); sw_df.index = bowl_ids
U = U.join(su_df)
U['player_name'] = [name_map.get(i,'Unknown') for i in U.index]
U['n_innings_or_spells'] = (bat_e['n'].reindex(U.index).fillna(0)+bowl_e['n'].reindex(U.index).fillna(0)).astype(int)
nm = pd.concat([bat_elig.set_index('player_id')['n_matches'], bowl_elig.set_index('player_id')['n_matches']],axis=1)
U['n_matches'] = nm.max(axis=1).reindex(U.index)
U['n_scenarios_active'] = pd.concat([bat_elig.set_index('player_id')['n_scenarios_active'], bowl_elig.set_index('player_id')['n_scenarios_active']],axis=1).max(axis=1).reindex(U.index)
cons = pd.concat([bat_elig.set_index('player_id')['consistency'], bowl_elig.set_index('player_id')['consistency']],axis=1).mean(axis=1)
U['consistency'] = cons.reindex(U.index)
rw = pd.concat([bat_elig.set_index('player_id')['recent_weak'], bowl_elig.set_index('player_id')['recent_weak']],axis=1).fillna(False).any(axis=1)
U['recent_weak'] = rw.reindex(U.index)
U = U.reset_index().rename(columns={'index':'player_id'}).sort_values('wpae_combined',ascending=False).reset_index(drop=True)
U['unified_rank'] = np.arange(1,len(U)+1)
U['stability_flag'] = (U['unified_rank']<=10)&(U['p_top10']<0.30)
top10, top25 = U.head(10).copy(), U.head(25).copy()
print("\n  UNIFIED TOP 10:")
for _,r in top10.iterrows():
    print(f"  {r.unified_rank:2d}. {r.player_name:24s} [{r.role:10s}] score={r.wpae_combined:+.3f} bat={r.bat_wpae:+.3f} bowl={r.bowl_wpae:+.3f} P(top10)={r.p_top10:.2f} med_rank={r.rank_median:.0f}{' UNSTABLE' if r.stability_flag else ''}")

# role lists for Unstop dropdowns (same model)
for e_df, s_df, nm_, col in ((bat_elig, sb_df,'bat','wpae_bat_shrunk'),(bowl_elig, sw_df,'bowl','wpae_bowl_shrunk')):
    pass
bat_elig = bat_elig.merge(sb_df.reset_index().rename(columns={'index':'player_id'}), on='player_id')
bowl_elig = bowl_elig.merge(sw_df.reset_index().rename(columns={'index':'player_id'}), on='player_id')
bat_elig['player_name'] = bat_elig['player_id'].map(name_map); bowl_elig['player_name'] = bowl_elig['player_id'].map(name_map)
bat_elig = bat_elig.sort_values('wpae_bat_shrunk',ascending=False).reset_index(drop=True)
bowl_elig = bowl_elig.sort_values('wpae_bowl_shrunk',ascending=False).reset_index(drop=True)
bat_top = bat_elig.head(10).copy(); bowl_top = bowl_elig.head(10).copy()
bat_top['rank']=range(1,11); bowl_top['rank']=range(1,11)
print("\n  TOP 10 BATTERS:"); [print(f"   {r['rank']:2d}. {r.player_name} [{r.player_id}] {r.wpae_bat_shrunk:+.3f} P(top10 bat)={r.p_top10:.2f}") for _,r in bat_top.iterrows()]
print("  TOP 10 BOWLERS:"); [print(f"   {r['rank']:2d}. {r.player_name} [{r.player_id}] {r.wpae_bowl_shrunk:+.3f} P(top10 bowl)={r.p_top10:.2f}") for _,r in bowl_top.iterrows()]

# ---------- sensitivity ----------
print("\n--- Sensitivity ---")
rows = []
def sens(label, mi, pid, val, balls, C0, thr, base_e):
    variants = {'base':dict(), 'no_LI_uncapped':dict(val=val.replace('_adj','_sum')), 'no_opposition_adj':dict(val=val.replace('_adj','_li_sum_capped')), 'halflife_6m':dict(hl=6),
                'halflife_24m':dict(hl=24), 'no_recency':dict(hl=None), 'no_shrinkage':dict(C=0.0), 'shrinkage_x2':dict(C=2*C0)}
    base_top = set(base_e.sort_values('shrunk',ascending=False).head(10).index)
    for name, kw in variants.items():
        a = rating(mi,pid,kw.get('val',val),balls,kw.get('hl',12),kw.get('C',C0))
        e = eligible(a,thr,is_bat=(label=='batters')); common = base_e.index.intersection(e.index)
        rho = spearmanr(base_e.loc[common,'shrunk'], e.loc[common,'shrunk'])[0]
        top = set(e.sort_values('shrunk',ascending=False).head(10).index)
        rows.append({'group':label,'variant':name,'spearman_vs_base':round(rho,3),'top10_overlap':len(base_top&top)})
    for k,(bt,wt) in THR.items():
        e = eligible(rating(mi,pid,val,balls,12,C0), bt if label=='batters' else wt, is_bat=(label=='batters'))
        top = set(e.sort_values('shrunk',ascending=False).head(10).index)
        rows.append({'group':label,'variant':f'threshold_{k}','spearman_vs_base':np.nan,'top10_overlap':len(base_top&top)})
sens('batters',bat_mi,'striker_id',BV,BB,C_bat,THR['medium'][0],bat_e)
sens('bowlers',bowl_mi,'bowler_id',WV,WB,C_bowl,THR['medium'][1],bowl_e)
sens_df = pd.DataFrame(rows); print(sens_df.to_string(index=False)); sens_df.to_csv(RES/'sensitivity_analysis.csv',index=False)

# ---------- effect of the Pakistan exclusion (point estimates, no exclusion) ----------
be_all = eligible(bat_all,THR['medium'][0],excl=False,is_bat=True); we_all = eligible(bowl_all,THR['medium'][1],excl=False,is_bat=False)
U0 = unify(be_all['shrunk'], we_all['shrunk']).sort_values('wpae_combined',ascending=False)
U0['rank_without_exclusion'] = np.arange(1,len(U0)+1)
old_top = U0.head(10); n_pak_old = int(old_top.index.isin(PAK).sum())
entered = top10[~top10['player_id'].isin(old_top.index)][['unified_rank','player_id','player_name']].copy()
entered['rank_if_no_exclusion'] = entered['player_id'].map(U0['rank_without_exclusion'])
entered.to_csv(RES/'entered_due_to_exclusion.csv', index=False)
print(f"\n  Without the exclusion, {n_pak_old} of the old top 10 were Pakistan players.")
print("  Entered the top 10 because of the exclusion:"); print(entered.to_string(index=False) if len(entered) else "  none")
json.dump({'n_pakistan_in_top10_without_exclusion':n_pak_old}, open(RES/'exclusion_effect.json','w'))

# ---------- assertions + save ----------
for nme, df_ in (('unified',U),('top25',top25),('top10',top10),('bat top',bat_top),('bowl top',bowl_top),('bat elig',bat_elig),('bowl elig',bowl_elig)):
    assert not df_['player_id'].isin(PAK).any(), f'PAK in {nme}'
for nme, df_ in (('top10',top10),('bat top',bat_top),('bowl top',bowl_top)):
    assert df_['player_id'].nunique()==10, f'dup in {nme}'
print("\n  ASSERTION PASSED: no excluded player_id in unified / top25 / top10 / role lists / eligible tables; no duplicates")
U.to_csv(RES/'top_unified_full.csv',index=False); top25.to_csv(RES/'top25_unified.csv',index=False); top10.to_csv(RES/'top10_unified.csv',index=False)
bat_top.to_csv(RES/'top10_batters_unstop.csv',index=False); bowl_top.to_csv(RES/'top10_bowlers_unstop.csv',index=False)
bat_elig.to_csv(RES/'bat_eligible_ranked.csv',index=False); bowl_elig.to_csv(RES/'bowl_eligible_ranked.csv',index=False)
json.dump({'C_bat':round(C_bat,3),'C_bowl':round(C_bowl,3)}, open(RES/'shrinkage_constants.json','w'), indent=2)

# ---------- rank stability figures ----------
def box(R, ids, order_ids, title, fn, color):
    pos = {p:i for i,p in enumerate(ids)}
    data = [R[:,pos[p]] for p in order_ids]
    fig, ax = plt.subplots(figsize=(12,5))
    ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor=color,alpha=.5), medianprops=dict(color='red',lw=2))
    ax.set_xticks(range(1,len(order_ids)+1)); ax.set_xticklabels([name_map.get(p,str(p))[:15] for p in order_ids], rotation=45, ha='right', fontsize=8)
    ax.axhline(10.5,color='red',ls='--',alpha=.6,label='Top-10 cutoff'); ax.set_ylabel('Bootstrap rank'); ax.set_title(title)
    ax.set_ylim(0, 45); ax.legend(); ax.grid(axis='y',alpha=.3); fig.tight_layout(); fig.savefig(FIG/fn,dpi=150); plt.close(fig)
box(R_u, uid, list(U.head(15)['player_id']), f'Rank stability - unified top 15 ({N_BOOT} match-level bootstrap resamples)', 'rank_stability_unified.png','seagreen')
box(R_b, bat_ids, list(bat_elig.head(15)['player_id']), f'Rank stability - top 15 batters ({N_BOOT} resamples)', 'rank_stability_bat.png','steelblue')
box(R_w, bowl_ids, list(bowl_elig.head(15)['player_id']), f'Rank stability - top 15 bowlers ({N_BOOT} resamples)', 'rank_stability_bowl.png','darkorange')
bat_mi.to_parquet(CLEAN/'bat_match_innings_adj.parquet', index=False); bowl_mi.to_parquet(CLEAN/'bowl_match_innings_adj.parquet', index=False)
print("\nSTEP 5 (v2) COMPLETE")
