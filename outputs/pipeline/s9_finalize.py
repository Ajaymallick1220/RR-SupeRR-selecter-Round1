"""
s9_finalize.py - evidence tables, dossiers, profiles, extra figures (mode 'data');
report appendices, final submission block, notebook, archive of stale files and real verification (mode 'docx').
SupeRR Selector S2 (v7). Seed 42.
"""
import sys, warnings, pathlib, json, shutil, re, datetime
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def ordn(x):
    n = int(round(x)); suf = 'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th'); return f'{n}{suf}'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'data'
WORK = pathlib.Path(r'D:\SupeRR_round_1'); OUT = WORK/'outputs'
CLEAN = OUT/'data_clean'; RES = OUT/'results'; FIG = OUT/'figures'; PIPE = OUT/'pipeline'
excl = pd.read_csv(RES/'excluded_pakistan_players.csv'); PAK = set(excl['player_id'])
ctpm = pd.read_csv(WORK/'competition_team_player_mapping.csv'); name_map = dict(zip(ctpm['player_id'], ctpm['player_name']))
teams = pd.read_csv(WORK/'teams.csv').drop_duplicates('team_id'); team_name = dict(zip(teams['team_id'], teams['team_name']))
SCEN = {'scen_A':'A chasing pressure','scen_B':'B setting under pressure','scen_C':'C knockout/final','scen_D':'D key moments',
        'scen_E':'E big ask','scen_F':'F collapse and rebuild','scen_G':'G high-scoring pitch','scen_H':'H turning point','scen_I':'I set-batter/bounce-back'}
PH = {'powerplay':'powerplay','middle':'middle overs','death':'death overs'}

top10 = pd.read_csv(RES/'top10_unified.csv'); top25 = pd.read_csv(RES/'top25_unified.csv'); full_u = pd.read_csv(RES/'top_unified_full.csv')
bat_el = pd.read_csv(RES/'bat_eligible_ranked.csv'); bowl_el = pd.read_csv(RES/'bowl_eligible_ranked.csv')
bat_top = pd.read_csv(RES/'top10_batters_unstop.csv'); bowl_top = pd.read_csv(RES/'top10_bowlers_unstop.csv')
C = json.load(open(RES/'shrinkage_constants.json')); C_bat, C_bowl = C['C_bat'], C['C_bowl']

if MODE == 'data':
    print("="*70); print("S9 DATA - turning points, bounce-back, dossiers, profiles, figures"); print("="*70)
    legal = pd.read_parquet(CLEAN/'deliveries_wpae.parquet'); legal['match_date'] = pd.to_datetime(legal['match_date'])
    legal = legal.sort_values(['match_id','innings','legal_ball_seq'])
    bat_mi = pd.read_parquet(CLEAN/'bat_match_innings_adj.parquet'); bowl_mi = pd.read_parquet(CLEAN/'bowl_match_innings_adj.parquet')
    for d in (bat_mi, bowl_mi): d['match_date'] = pd.to_datetime(d['match_date'])
    bat_prof = pd.read_csv(RES/'bat_skill_profiles.csv'); bowl_prof = pd.read_csv(RES/'bowl_skill_profiles.csv')

    # ---------- half-life test (true out-of-time) ----------
    print("\n--- Half-life test: rating from data before 2025-01-01, scored on 2025+ ---")
    cut = pd.Timestamp('2025-01-01'); rows = []
    def hl_rows(mi, pid, val, Cc, label):
        tr = mi[mi.match_date < cut]; te = mi[mi.match_date >= cut]
        tm = te.groupby(pid)[val].agg(['mean','count']); tm = tm[tm['count'] >= 5]
        for hl in (6, 12, 18, 24):
            lam = np.log(2)/(hl*30.44); w = np.exp(-lam*(cut-tr.match_date).dt.days.clip(lower=0))
            g = pd.DataFrame({'p':tr[pid],'v':tr[val]*w,'w':w}).groupby('p').agg(vw=('v','sum'),ws=('w','sum'),n=('w','size'))
            g = g[(g.n >= 10)]; g['r'] = g.vw/g.ws*g.n/(g.n+Cc)
            j = g.join(tm, how='inner'); rho, p = spearmanr(j['r'], j['mean'])
            rows.append({'group':label,'half_life_months':hl,'n_players':len(j),'spearman':round(rho,3),'p_value':round(p,4)})
    hl_rows(bat_mi,'striker_id','wpae_bat_adj',C_bat,'batters'); hl_rows(bowl_mi,'bowler_id','wpae_bowl_adj',C_bowl,'bowlers')
    hl = pd.DataFrame(rows); hl.to_csv(RES/'halflife_test.csv', index=False); print(hl.to_string(index=False))

    # ---------- turning points (12-ball blocks, |delta WP| >= 15 pp, share >= 30%) ----------
    print("\n--- Turning points ---")
    legal['blk'] = legal['legal_ball_seq']//12
    blk = legal.groupby(['match_id','innings','blk']).agg(delta=('wpa_batter','sum'), wp_start=('wp_before','first'), wp_end=('wp_after','last'),
            b0=('legal_ball_seq','min'), b1=('legal_ball_seq','max'), date=('match_date','first'), comp=('comp_name','first'),
            bat_team=('batting_team_id','first'), bowl_team=('bowling_team_id','first')).reset_index()
    big = blk[blk['delta'].abs() >= 0.15].copy(); print(f"  blocks with |dWP|>=15pp: {len(big)} of {len(blk)}")
    sub = legal.merge(big[['match_id','innings','blk','delta']], on=['match_id','innings','blk'])
    sub['s'] = np.sign(sub['delta'])
    bat_c = sub[sub['s']>0].groupby(['match_id','innings','blk','striker_id'])['wpa_batter'].sum().reset_index().rename(columns={'striker_id':'player_id','wpa_batter':'contrib'}); bat_c['role']='batter'
    bow_c = sub[sub['s']<0].groupby(['match_id','innings','blk','bowler_id'])['wpa_bowler'].sum().reset_index().rename(columns={'bowler_id':'player_id','wpa_bowler':'contrib'}); bow_c['role']='bowler'
    tp = pd.concat([bat_c, bow_c]).merge(big, on=['match_id','innings','blk'])
    tp['share'] = tp['contrib']/tp['delta'].abs(); tp = tp[(tp['share'] >= 0.30) & ~tp['player_id'].isin(PAK)]
    tp['player_name'] = tp['player_id'].map(name_map); tp['favoured_side'] = np.where(tp['delta']>0,'batting side','bowling side')
    tp['opponent'] = np.where(tp['role']=='batter', tp['bowl_team'].map(team_name), tp['bat_team'].map(team_name))
    tp = tp[['match_id','date','comp','innings','b0','b1','wp_start','wp_end','delta','favoured_side','player_id','player_name','role','contrib','share','opponent']]
    tp.rename(columns={'b0':'first_legal_ball','b1':'last_legal_ball','delta':'delta_wp_batting_side'}, inplace=True)
    tp.to_csv(RES/'turning_points.csv', index=False); print(f"  attributions kept (non-excluded, share>=30%): {len(tp)}")
    tp_cnt = tp.groupby('player_id').agg(n_turning=('share','size'), sum_share=('share','sum'))

    # ---------- bounce-back ----------
    print("\n--- Bounce-back after heavy overs ---")
    ov = legal.groupby(['match_id','innings','bowler_id','over_0based']).agg(runs=('total_del_runs','sum'), balls=('is_legal','sum'),
            wpae=('wpae_bowl_li','sum'), wk=('is_wicket_bowl','sum')).reset_index().sort_values(['match_id','innings','bowler_id','over_0based'])
    ov['heavy'] = ov['runs'] >= 15; g = ov.groupby(['match_id','innings','bowler_id'])
    ov['next_runs'] = g['runs'].shift(-1); ov['next_balls'] = g['balls'].shift(-1); ov['next_wpae'] = g['wpae'].shift(-1); ov['next_wk'] = g['wk'].shift(-1)
    elig_b = set(bowl_el['player_id'])
    ovb = ov[ov['bowler_id'].isin(elig_b) & ~ov['bowler_id'].isin(PAK) & (ov['balls']>=4)]
    h = ovb[ovb['heavy'] & ovb['next_runs'].notna()].groupby('bowler_id').agg(n_heavy=('runs','size'), next_over_runs=('next_runs','sum'),
            next_over_balls=('next_balls','sum'), next_over_wpae=('next_wpae','mean'), next_over_wkts=('next_wk','sum'))
    a = ovb.groupby('bowler_id').agg(all_overs=('runs','size'), all_runs=('runs','sum'), all_balls=('balls','sum'), all_over_wpae=('wpae','mean'))
    bb = h.join(a); bb['next_over_economy'] = 6*bb['next_over_runs']/bb['next_over_balls']; bb['all_over_economy'] = 6*bb['all_runs']/bb['all_balls']
    bb['wpae_change_vs_usual'] = bb['next_over_wpae']-bb['all_over_wpae']; bb = bb[bb['n_heavy']>=3].reset_index()
    bb['player_name'] = bb['bowler_id'].map(name_map); bb = bb.rename(columns={'bowler_id':'player_id'})
    bb.to_csv(RES/'bounce_back.csv', index=False); print(f"  bowlers with >=3 heavy overs followed by another over: {len(bb)}")
    bbi = bb.set_index('player_id')

    # ---------- dossiers ----------
    print("\n--- Dossiers ---")
    fin = list(full_u.head(30)['player_id']); fin = [p for p in fin if p not in PAK]
    bat_ids = set(bat_el['player_id']); bowl_ids = set(bowl_el['player_id'])
    def scen_tags(r, thresh=6): return ', '.join(SCEN[s] for s in SCEN if r.get(s,0) >= thresh) or 'no tagged scenario'
    ev_rows = []
    for role, pid, ids, mi, adj, wcol, swing, ctx in (('batter','striker_id',bat_ids,bat_mi,'wpae_bat_adj','is_wicket_bat','wpa_batter','bowling_team_id'),
                                                      ('bowler','bowler_id',bowl_ids,bowl_mi,'wpae_bowl_adj','is_wicket_bowl','wpa_bowler','batting_team_id')):
        want = [p for p in fin if p in ids]; g = legal[legal[pid].isin(want)]
        aggd = {'date':('match_date','first'),'comp':('comp_name','first'),'balls':('is_legal','sum'),'wkts':(wcol,'sum'),'wpa':(swing,'sum'),
                'opp':(ctx,'first'),'phase':('phase',lambda x:x.mode().iloc[0]),'balls_left':('legal_balls_remaining','first'),
                'runs_req':('runs_required','first'),'wp_first':('wp_before','first'),'wp_last':('wp_after','last')}
        for s in SCEN: aggd[s] = (s,'sum')
        if role=='batter': aggd['runs'] = ('runs_bat','sum')
        else: aggd['runs'] = ('total_del_runs','sum')
        ev = g.groupby([pid,'match_id','innings']).agg(**aggd).reset_index().merge(mi[[pid,'match_id','innings',adj]], on=[pid,'match_id','innings'])
        for p, d in ev.groupby(pid):
            d = d.sort_values(adj); sel = pd.concat([d.tail(3).iloc[::-1].assign(kind='best'), d.head(3).assign(kind='worst')])
            for _, r in sel.iterrows():
                wps = r['wp_first'] if role=='batter' else 1-r['wp_first']; wpe = r['wp_last'] if role=='batter' else 1-r['wp_last']
                ev_rows.append({'player_id':p,'player_name':name_map.get(p),'role':role,'kind':r['kind'],'date':r['date'].date(),'competition':r['comp'],
                    'opponent':team_name.get(r['opp']),'innings':int(r['innings']),'balls':int(r['balls']),'runs':int(r['runs']),'wickets':int(r['wkts']),
                    'phase':r['phase'],'balls_left_at_start':r['balls_left'],'runs_required_at_start':r['runs_req'],
                    'own_side_win_prob_start':round(wps,3),'own_side_win_prob_end':round(wpe,3),'wp_swing_from_his_balls':round(r['wpa'],3),
                    'wpae_adjusted':round(r[adj],3),'scenario_tags':scen_tags(r)})
    doss = pd.DataFrame(ev_rows); assert not doss['player_id'].isin(PAK).any(); doss.to_csv(RES/'finalist_dossiers.csv', index=False)
    print(f"  dossier events: {len(doss)} for {doss.player_id.nunique()} finalists")

    # ---------- summaries: windows, scenario percentiles, peers ----------
    cut12, cut24 = pd.Timestamp('2026-11-01')-pd.DateOffset(months=12), pd.Timestamp('2026-11-01')-pd.DateOffset(months=24)
    def win(mi, pid, col, p, Cc):
        d = mi[mi[pid]==p]; out = {}
        for nm_, m in (('full',d),('last24',d[d.match_date>=cut24]),('last12',d[d.match_date>=cut12])):
            n = len(m); out[nm_] = (round(m[col].mean()*n/(n+Cc),3) if n else np.nan, n)
        return out
    def scen_pct(pidcol, valcol, ids):
        res = {}
        for s in SCEN:
            f = legal[(legal[s]==1) & legal[pidcol].isin(ids)].groupby(pidcol)[valcol].agg(['sum','count']); f = f[f['count']>=30]
            f['per100'] = 100*f['sum']/f['count']; f['pct'] = f['per100'].rank(pct=True)*100; res[s] = f
        return res
    sp_bat = scen_pct('striker_id','wpae_bat_li',bat_ids); sp_bowl = scen_pct('bowler_id','wpae_bowl_li',bowl_ids)
    bat_el['bat_role'] = bat_el['bat_role'].fillna('unknown')
    bat_el['peer_pct'] = bat_el.groupby('bat_role')['wpae_bat_shrunk'].rank(pct=True)*100
    bowl_el['peer_pct'] = bowl_el.groupby('pace_or_spin')['wpae_bowl_shrunk'].rank(pct=True)*100
    be = bat_el.set_index('player_id'); we = bowl_el.set_index('player_id')
    summ = []
    for p in fin:
        r = full_u[full_u.player_id==p].iloc[0]; d = {'player_id':p,'player_name':name_map.get(p),'unified_rank':int(r.unified_rank),'role':r.role,
            'wpae_combined':round(r.wpae_combined,3),'bat_wpae':round(r.bat_wpae,3),'bowl_wpae':round(r.bowl_wpae,3),
            'p_top10':round(r.p_top10,2),'rank_median':r.rank_median,'rank_p80_conservative':r.rank_p80,'score_p20':round(r.score_p20,3)}
        L = legal[(legal.striker_id==p)|(legal.bowler_id==p)]
        d.update(matches=L['match_id'].nunique(), seasons=L['season_year'].nunique(), competitions=L['comp_name'].nunique(),
                 first_match=L['match_date'].min().date(), last_match=L['match_date'].max().date())
        if p in bat_ids:
            d['bat_peer_group'] = be.loc[p,'bat_role']; d['bat_peer_pct'] = round(be.loc[p,'peer_pct']); w = win(bat_mi,'striker_id','wpae_bat_adj',p,C_bat)
            for k,(v,n) in w.items(): d[f'bat_{k}_shrunk'] = v; d[f'bat_{k}_n'] = n
            bl = [(sp_bat[s].loc[p,'pct'], SCEN[s]) for s in SCEN if p in sp_bat[s].index]; bl.sort()
            d['bat_weakest_scenarios'] = '; '.join(f'{n} ({ordn(q)} percentile)' for q,n in bl[:2]); d['bat_strongest_scenarios'] = '; '.join(f'{n} ({ordn(q)} percentile)' for q,n in bl[::-1][:2])
        if p in bowl_ids:
            d['bowl_peer_group'] = we.loc[p,'pace_or_spin']; d['bowl_peer_pct'] = round(we.loc[p,'peer_pct']); w = win(bowl_mi,'bowler_id','wpae_bowl_adj',p,C_bowl)
            for k,(v,n) in w.items(): d[f'bowl_{k}_shrunk'] = v; d[f'bowl_{k}_n'] = n
            bl = [(sp_bowl[s].loc[p,'pct'], SCEN[s]) for s in SCEN if p in sp_bowl[s].index]; bl.sort()
            d['bowl_weakest_scenarios'] = '; '.join(f'{n} ({ordn(q)} percentile)' for q,n in bl[:2]); d['bowl_strongest_scenarios'] = '; '.join(f'{n} ({ordn(q)} percentile)' for q,n in bl[::-1][:2])
        d['turning_point_credits'] = int(tp_cnt.n_turning.get(p,0))
        if p in bbi.index: d['heavy_overs_followed'] = int(bbi.loc[p,'n_heavy']); d['next_over_economy'] = round(bbi.loc[p,'next_over_economy'],2); d['usual_economy'] = round(bbi.loc[p,'all_over_economy'],2)
        summ.append(d)
    summ = pd.DataFrame(summ); summ.to_csv(RES/'finalist_summary.csv', index=False); assert not summ.player_id.isin(PAK).any()

    # ---------- cricket-language profiles for the unified top 10 ----------
    def profile(p):
        s = summ[summ.player_id==p].iloc[0]; r = top10[top10.player_id==p].iloc[0]; nm = s.player_name; T = []
        T.append(f"{nm} (player_id {p}) is ranked {int(r.unified_rank)} in the combined list with a score of {r.wpae_combined:+.3f} win-probability points per appearance "
                 f"(batting {r.bat_wpae:+.3f}, bowling {r.bowl_wpae:+.3f}). The evidence base is {int(s.matches)} matches in {int(s.seasons)} season(s) across {int(s.competitions)} competition(s), "
                 f"last played {s.last_match}.")
        if p in bat_ids:
            b = be.loc[p]; T.append(f"As a batter he is compared with other {b.bat_role}s: his shrunk rating is {b.wpae_bat_shrunk:+.3f} per innings, around the {ordn(s.bat_peer_pct)} percentile of eligible batters in that role.")
            bp = bat_prof[(bat_prof.striker_id==p)&(bat_prof.balls_faced>=60)].copy()
            if len(bp):
                x = bp.sort_values('sr_vs_pg').iloc[-1]; T.append(f"His best batting phase against his peer group is the {PH.get(x.phase,x.phase)}: {x.sr:.0f} runs per 100 balls over {int(x.balls_faced)} balls against {x.sr_pg:.0f} for comparable batters (all match situations).")
        if p in bowl_ids:
            w = we.loc[p]; T.append(f"As a bowler he is compared with other {str(w.pace_or_spin).lower()} bowlers: his shrunk rating is {w.wpae_bowl_shrunk:+.3f} per spell, around the {ordn(s.bowl_peer_pct)} percentile of eligible {str(w.pace_or_spin).lower()} bowlers.")
            bp = bowl_prof[(bowl_prof.bowler_id==p)&(bowl_prof.balls>=60)].copy()
            if len(bp):
                x = bp.sort_values('eco_vs_pg').iloc[0]
                T.append(f"His best bowling phase against peers is the {PH.get(x.phase,x.phase)}: {x.economy:.2f} runs an over over {int(x.balls)} balls against {x.eco_pg:.2f} for {str(w.pace_or_spin).lower()} bowlers in that phase, with a dot-ball rate of {x.dot_pct:.0f}% against {x.dot_pct_pg:.0f}% (all match situations).")
                y = bp[bp.phase=='death']
                if len(y) and y.iloc[0].yorker_pct > 0: T.append(f"In the death overs {y.iloc[0].yorker_pct:.1f}% of his deliveries are tagged as yorkers.")
        for role in ('batter','bowler'):
            d = doss[(doss.player_id==p)&(doss.role==role)]
            if len(d):
                b_, w_ = d[d.kind=='best'].iloc[0], d[d.kind=='worst'].iloc[0]
                def line(e, tag):
                    do = f"scored {e.runs} off {e.balls}" if role=='batter' else f"took {e.wickets} for {e.runs} in {e.balls} balls"
                    return (f"{tag}: {e.date} against {e.opponent} ({e.competition}), he {do} in the {PH.get(e.phase,e.phase)}; his side's win probability moved from {e.own_side_win_prob_start:.0%} to {e.own_side_win_prob_end:.0%} over his involvement "
                            f"(his own balls contributed {e.wp_swing_from_his_balls:+.2f}); tags: {e.scenario_tags}.")
                T.append(line(b_, f"Best {role} performance")); T.append(line(w_, f"Weakest {role} performance"))
        for k in ('bat','bowl'):
            if f'{k}_full_shrunk' in s and not pd.isna(s.get(f'{k}_full_shrunk')):
                f_, a_, l_ = s[f'{k}_full_shrunk'], s[f'{k}_last24_shrunk'], s[f'{k}_last12_shrunk']; nl = int(s[f'{k}_last12_n'])
                T.append(f"{'Batting' if k=='bat' else 'Bowling'} trend (shrunk, per appearance): full period {f_:+.3f}, last 24 months {a_:+.3f}, last 12 months {l_:+.3f} ({nl} appearances in the last 12 months)" + ("; the last-12-month figure rests on a small sample." if nl < 8 else "."))
        if 'bat_strongest_scenarios' in s and isinstance(s.get('bat_strongest_scenarios'),str): T.append(f"Strongest batting scenarios: {s.bat_strongest_scenarios}; weakest: {s.bat_weakest_scenarios}.")
        if 'bowl_strongest_scenarios' in s and isinstance(s.get('bowl_strongest_scenarios'),str): T.append(f"Strongest bowling scenarios: {s.bowl_strongest_scenarios}; weakest: {s.bowl_weakest_scenarios}.")
        if p in bbi.index: T.append(f"After {int(s.heavy_overs_followed)} overs costing 15+ runs he bowled again at {s.next_over_economy:.2f} runs an over against his usual {s.usual_economy:.2f}.")
        if s.turning_point_credits: T.append(f"He is credited (at least 30% of the swing) in {int(s.turning_point_credits)} twelve-ball turning-point windows.")
        T.append(f"Stability: he finished in the combined top 10 in {r.p_top10:.0%} of 300 match-level bootstrap resamples (median rank {r.rank_median:.0f}, conservative 80th-percentile rank {s.rank_p80_conservative:.0f})" + (" - flagged unstable." if r.stability_flag else "."))
        return T
    profs = {int(p): profile(p) for p in top10['player_id']}
    json.dump(profs, open(RES/'finalist_profiles.json','w'), indent=1)
    for p, t in list(profs.items())[:2]: print(f"\n  PROFILE {p}:"); [print('   -', x) for x in t]

    # ---------- extra figures ----------
    oot = pd.read_csv(RES/'oot_validation.csv'); fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(oot['split'], oot['spearman_rho'], color=['steelblue' if 'Bowl' not in s else 'darkorange' for s in oot['split']])
    for i,(r_,n_) in enumerate(zip(oot['spearman_rho'], oot['n_players'])): ax.text(r_+0.003, i, f'rho={r_:.2f} (n={n_})', va='center', fontsize=8)
    ax.axvline(0, color='k', lw=.8); ax.axvline(0.3, color='red', ls=':', label='0.3 reference'); ax.set_xlabel('Spearman correlation (train rating vs later performance)')
    ax.set_title('Out-of-time validation'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'out_of_time_result.png', dpi=150); plt.close(fig)

    L = legal[~legal.striker_id.isin(PAK)].copy(); L['bin'] = pd.cut(L['batter_balls_so_far'], [-1,5,11,17,23,1000], labels=['1-6','7-12','13-18','19-24','25+'])
    t10b = set(bat_top['player_id']); fig, ax = plt.subplots(figsize=(8,5))
    for lab, m, st in (('Arrived after 3+ wickets in first 8 overs (all batters)', L.scen_F==1, '-o'), ('All other situations (all batters)', L.scen_F==0, '--s'),
                       ('Top-10 batters, after early collapse', (L.scen_F==1)&L.striker_id.isin(t10b), '-^')):
        g = L[m].groupby('bin').agg(r=('runs_bat','sum'), b=('is_legal','sum')); ax.plot(range(len(g)), 100*g.r/g.b, st, label=f'{lab} (n={int(g.b.sum()):,} balls)')
    ax.set_xticks(range(5)); ax.set_xticklabels(['1-6','7-12','13-18','19-24','25+']); ax.set_xlabel('Balls faced in the innings so far'); ax.set_ylabel('Strike rate (runs per 100 balls)')
    ax.set_title('Collapse-and-rebuild: strike rate by balls faced'); ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIG/'collapse_rebuild_curves.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1,2, figsize=(12,5)); topids = set(top10['player_id'])
    for ax, df_, grp, col, ttl in ((axes[0], bat_el, 'bat_role','wpae_bat_shrunk','Batters by role'), (axes[1], bowl_el, 'pace_or_spin','wpae_bowl_shrunk','Bowlers: pace vs spin')):
        keys = sorted(df_[grp].dropna().unique()); ax.boxplot([df_[df_[grp]==k][col] for k in keys], labels=keys, showfliers=False)
        for i,k in enumerate(keys):
            d = df_[(df_[grp]==k)&df_['player_id'].isin(topids)]; ax.scatter(np.full(len(d), i+1), d[col], color='red', zorder=3, s=25, label='Unified top-10' if i==0 else None)
        ax.set_title(ttl); ax.set_ylabel('Shrunk WPAE per appearance'); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8)
    fig.suptitle('Like-with-like peer groups (eligible players, medium thresholds)'); fig.tight_layout(); fig.savefig(FIG/'peer_group_comparison.png', dpi=150); plt.close(fig)

    # ---------- Batter Phase Pressure Model (85-90+ balls under pressure per phase) ----------
    if (RES/'batter_phase_pressure_model_85b.csv').exists():
        ph85 = pd.read_csv(RES/'batter_phase_pressure_model_85b.csv')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, ph, col in zip(axes, ['powerplay', 'middle', 'death'], ['steelblue', 'seagreen', 'darkorange']):
            d = ph85[ph85['phase']==ph].head(10).iloc[::-1]
            ax.barh(d['player_name'].str[:16], d['wpae_per_100b_shrunk'], color=col, alpha=0.85)
            for idx_i, (_, r_) in enumerate(d.iterrows()):
                ax.text(r_['wpae_per_100b_shrunk'] + 0.03, idx_i, f"{int(r_['p_balls'])}b (SR {r_['sr']:.0f})", va='center', fontsize=7)
            ax.set_title(f"{ph.capitalize()} (>=85 pressure balls)")
            ax.set_xlabel("Shrunk WPAE per 100 pressure balls")
            ax.set_xlim(0, d['wpae_per_100b_shrunk'].max() * 1.32)
            ax.grid(axis='x', alpha=0.3)
        fig.suptitle("Batter Phase-Pressure Model (Min 85 Match-Situation Pressure Balls in Phase; Full-Member/IPL Only)", fontsize=11, fontweight='bold')
        fig.tight_layout()
        fig.savefig(FIG/'batter_phase_pressure_85b.png', dpi=150)
        plt.close(fig)

    print("\nS9 DATA COMPLETE")

elif MODE == 'docx':
    from docx import Document
    from docx.shared import Inches
    print("="*70); print("S9 DOCX - appendices, submission block, notebook, archive, verification"); print("="*70)
    profs = json.load(open(RES/'finalist_profiles.json')); hl = pd.read_csv(RES/'halflife_test.csv')
    ent = pd.read_csv(RES/'entered_due_to_exclusion.csv'); eff = json.load(open(RES/'exclusion_effect.json'))
    val = json.load(open(RES/'validation_summary.json')); sens = pd.read_csv(RES/'sensitivity_analysis.csv')
    tp = pd.read_csv(RES/'turning_points.csv'); bb = pd.read_csv(RES/'bounce_back.csv')

    # --- stale-file archive (outputs of the earlier pipeline) ---
    arch = OUT/'archive_old_pipeline'; arch.mkdir(exist_ok=True); cutoff = datetime.datetime(2026,11,13,11,0,0).timestamp(); keep = {'data_dictionary.csv','excluded_pakistan_players.csv'}
    moved = []
    for d in (RES, FIG):
        for f in d.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff and f.name not in keep: shutil.move(str(f), str(arch/f.name)); moved.append(f.name)
    for f in PIPE.glob('step*.py'): shutil.move(str(f), str(arch/f.name)); moved.append(f.name)
    inter = CLEAN/'intermediate'; inter.mkdir(exist_ok=True)
    for n in ('bat_career_raw.csv','bowl_career_raw.csv'):
        if (RES/n).exists(): shutil.move(str(RES/n), str(inter/n)); moved.append(n+' (-> data_clean/intermediate)')
    for n, idc in (('bat_skill_profiles.csv','striker_id'),('bowl_skill_profiles.csv','bowler_id')):
        d = pd.read_csv(RES/n); d = d[~d[idc].isin(PAK)]; d.to_csv(RES/n, index=False)
    print(f"  archived {len(moved)} stale files: {moved}")

    # --- final submission block (single source: the CSVs) ---
    lines = ["=== FINAL RANKINGS - SUPERR SELECTOR S2 (v7) ===", "", "UNIFIED TOP 10 (Batters + Bowlers + All-rounders):"]
    for _, r in top10.iterrows():
        lines.append(f"{int(r.unified_rank)}. {r.player_name} (player_id {int(r.player_id)}) [{r.role}] combined={r.wpae_combined:+.3f} batting_share={r.bat_wpae:+.3f} bowling_share={r.bowl_wpae:+.3f} P(top10)={r.p_top10:.2f}")
    lines += ["", "TOP 10 BATTERS (Unstop dropdown):"] + [f"{int(r['rank'])}. {r.player_name}" for _, r in bat_top.iterrows()]
    lines += ["", "TOP 10 BOWLERS (Unstop dropdown):"] + [f"{int(r['rank'])}. {r.player_name}" for _, r in bowl_top.iterrows()]
    (OUT/'final_submission.txt').write_text('\n'.join(lines), encoding='utf-8'); print('\n'.join(lines))

    # --- docx appendices ---
    doc = Document(OUT/'Methodology_Report.docx')
    doc.add_page_break(); doc.add_heading('Appendix A - Detailed finalist profiles (unified top 10)', 1)
    doc.add_paragraph(f"Effect of the exclusion: without it, {eff['n_pakistan_in_top10_without_exclusion']} of the combined top 10 would have been excluded players. "
                      + ("No player entered the top 10 because of the exclusion." if len(ent)==0 else "Entered because of the exclusion: " + '; '.join(f"{r.player_name} (player_id {r.player_id}, rank {r.rank_if_no_exclusion:.0f} without the exclusion)" for _, r in ent.iterrows()) + ".")
                      + " The ranking is a per-appearance rating, so the exclusion changes at most the lower places.")
    for _, r in top10.iterrows():
        doc.add_heading(f"{int(r.unified_rank)}. {r.player_name}", 2)
        for t in profs[str(int(r.player_id))]: doc.add_paragraph(t, style='List Bullet')
    doc.add_page_break(); doc.add_heading('Appendix B - Evidence tables', 1)
    def tbl(df_, cap):
        doc.add_paragraph(cap).runs[0].bold = True; t = doc.add_table(rows=1+len(df_), cols=len(df_.columns)); t.style = 'Table Grid'
        for i, c in enumerate(df_.columns): t.rows[0].cells[i].text = str(c)
        for ri, (_, row) in enumerate(df_.iterrows()):
            for ci, v in enumerate(row): t.rows[ri+1].cells[ci].text = f"{v:.3f}" if isinstance(v, float) else str(v)
    tbl(hl, 'Half-life test: rating built from data before 2025-01-01, correlation with 2025+ performance')
    best = hl.sort_values('spearman', ascending=False).groupby('group').head(1)
    doc.add_paragraph("Best half-life by this out-of-time test: " + '; '.join(f"{r.group} {int(r.half_life_months)} months (rho {r.spearman:.3f})" for _, r in best.iterrows()) +
                      ". All correlations are low and differences between half-lives are small relative to their sampling error, so the 12-month half-life was kept because recent performance is meant to lead the selection; the sensitivity table shows how much the top 10 moves under other choices.")
    tbl(sens, 'Sensitivity of the batter and bowler rankings (Spearman vs base; overlap with base top 10)')
    top_tp = tp.groupby(['player_id','player_name']).agg(windows=('share','size'), mean_share=('share','mean')).reset_index().sort_values('windows', ascending=False).head(10)
    tbl(top_tp, 'Players credited most often (share >= 30%) in 12-ball windows where win probability moved >= 15 points')
    tbl(bb.sort_values('wpae_change_vs_usual', ascending=False).head(10)[['player_id','player_name','n_heavy','next_over_economy','all_over_economy','wpae_change_vs_usual']], 'Bounce-back: best next-over response after overs costing 15+ runs (>= 3 such overs)')
    if (RES/'batter_phase_pressure_model_85b.csv').exists():
        ph85 = pd.read_csv(RES/'batter_phase_pressure_model_85b.csv')
        for ph in ['powerplay', 'middle', 'death']:
            d = ph85[ph85['phase']==ph][['striker_id','player_name','bat_role','p_balls','n_inn','sr','bdry_pct','dot_pct','wpae_per_100b_shrunk']].head(10).rename(
                columns={'striker_id':'player_id','p_balls':'pressure_balls','wpae_per_100b_shrunk':'shrunk_wpae_per_100b'})
            tbl(d, f'Phase-Pressure Batter Model ({ph.capitalize()}, minimum 85 match-situation pressure balls in phase, Full-Member/IPL only)')
    doc.add_page_break(); doc.add_heading('Appendix C - Figures', 1)
    for f in sorted(FIG.glob('*.png')):
        doc.add_paragraph(f.stem.replace('_',' ').capitalize()).runs[0].bold = True; doc.add_picture(str(f), width=Inches(5.8))
    doc.save(OUT/'Methodology_Report.docx'); print("  docx appendices added")

    # --- notebook ---
    import nbformat as nbf
    nb = nbf.v4.new_notebook(); cells = [nbf.v4.new_markdown_cell("# SupeRR Selector S2 - analysis notebook\nLoads the saved outputs of the pipeline (`s0` ... `s9`). Run the scripts in order first; see README.")]
    cells.append(nbf.v4.new_code_cell("import pandas as pd, json\nfrom IPython.display import Image, display\nR='../results/'; F='../figures/'\nex=pd.read_csv(R+'excluded_pakistan_players.csv'); print('excluded players:', len(ex))"))
    for t, f in (('Unified top 10','top10_unified.csv'),('Top 10 batters','top10_batters_unstop.csv'),('Top 10 bowlers','top10_bowlers_unstop.csv')):
        cells.append(nbf.v4.new_markdown_cell(f'## {t}')); cells.append(nbf.v4.new_code_cell(f"d=pd.read_csv(R+'{f}'); assert not d.player_id.isin(ex.player_id).any(); d.head(10)"))
    cells.append(nbf.v4.new_markdown_cell('## Validation')); cells.append(nbf.v4.new_code_cell("print(json.dumps(json.load(open(R+'validation_summary.json')), indent=1)); pd.read_csv(R+'sensitivity_analysis.csv')"))
    cells.append(nbf.v4.new_markdown_cell('## Figures')); cells.append(nbf.v4.new_code_cell("for f in ['rank_stability_unified.png','calibration_inn2.png','collapse_rebuild_curves.png','peer_group_comparison.png']: display(Image(F+f))"))
    nb['cells'] = cells; nbf.write(nb, PIPE/'analysis.ipynb')
    print('  notebook written (not executed here; run it in Jupyter after the pipeline scripts)')

if MODE == 'verify':
    from docx import Document
    top10 = pd.read_csv(RES/'top10_unified.csv'); bat_top = pd.read_csv(RES/'top10_batters_unstop.csv'); bowl_top = pd.read_csv(RES/'top10_bowlers_unstop.csv')
    # --- real verification ---
    print("\n--- VERIFICATION ---")
    names_excl = {str(n).strip().lower() for n in excl['player_name'].dropna()}
    bad = []
    for f in list(RES.glob('*.csv')) + [OUT/'final_submission.txt']:
        if f.name == 'excluded_pakistan_players.csv': continue
        if f.suffix == '.csv':
            d = pd.read_csv(f)
            for c in d.columns:
                if c in ('player_id','striker_id','bowler_id') and d[c].isin(PAK).any(): bad.append(f"{f.name}:{c}")
    assert not bad, f"excluded ids found in {bad}"; print("  OK: no excluded player_id in any results CSV")
    txt = '\n'.join(p.text for p in Document(OUT/'Methodology_Report.docx').paragraphs)
    for t in Document(OUT/'Methodology_Report.docx').tables:
        for row in t.rows: txt += '\n' + ' '.join(c.text for c in row.cells)
    low = txt.lower(); hits = [n for n in names_excl if n and re.search(r'\b'+re.escape(n)+r'\b', low)]
    assert not hits, f"excluded names in report: {hits}"; print("  OK: no excluded player name in the report text/tables")
    sub = (OUT/'final_submission.txt').read_text(encoding='utf-8'); hits = [n for n in names_excl if re.search(r'\b'+re.escape(n)+r'\b', sub.lower())]
    assert not hits; print("  OK: no excluded name in the submission block")
    # identical rankings: report vs CSV vs block
    def seq_from(text, start, n, pat): 
        i = text.index(start); return re.findall(pat, text[i:i+4000])[:n]
    csv_u = [f"{int(r.unified_rank)}. {r.player_name}" for _, r in top10.iterrows()]
    rep_u = [f"{a}. {b.strip()}" for a, b in re.findall(r'^\s*(\d+)\.\s+(.*?)\s+\[', txt[txt.index('UNIFIED TOP 10'):], flags=re.M)[:10]]
    blk_u = [f"{a}. {b.strip()}" for a, b in re.findall(r'^(\d+)\.\s+(.*?)\s+\(player_id', sub, flags=re.M)[:10]]
    assert csv_u == rep_u == blk_u, f"unified mismatch\nCSV {csv_u}\nREP {rep_u}\nBLK {blk_u}"; print("  OK: unified top 10 identical in CSV, report and block")
    for lab, d in (('TOP 10 BATTERS', bat_top), ('TOP 10 BOWLERS', bowl_top)):
        csv_l = [f"{int(r['rank'])}. {r.player_name}" for _, r in d.iterrows()]
        rep_l = [f"{a}. {b.strip()}" for a, b in re.findall(r'^\s*(\d+)\.\s+(.*)$', txt[txt.index(lab):], flags=re.M)[:10]]
        blk_l = [f"{a}. {b.strip()}" for a, b in re.findall(r'^(\d+)\.\s+(.*)$', sub[sub.index(lab):], flags=re.M)[:10]]
        assert csv_l == rep_l == blk_l, f"{lab} mismatch\nCSV {csv_l}\nREP {rep_l}\nBLK {blk_l}"; print(f"  OK: {lab.lower()} identical in CSV, report and block")
    for nme, d in (('unified', top10), ('batters', bat_top), ('bowlers', bowl_top)): assert d.player_id.is_unique and len(d)==10
    print("  OK: no repeated player in any list;", f"excluded players: {len(PAK)}")
    print("\nS9 DOCX COMPLETE")

