# 🏏 Rajasthan Royals — SupeRR Selector Season 2 (Round 1)
## Identifying the Most Reliable T20 Players Under Pressure (IPL & Men's T20Is, 2022–2026)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](SupeRR_Selector_Round1_Analysis.ipynb)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.1%2B-green.svg)](https://scikit-learn.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified%20(Seed%2042)-brightgreen.svg)]()
[![License: Confidential / Proprietary](https://img.shields.io/badge/Data%20Policy-Confidential%20NDA-red.svg)]()

---

## 📌 Executive Summary

This repository houses the end-to-end quantitative analytics engine developed for the **Rajasthan Royals SupeRR Selector Season 2 (Round 1)** challenge. The primary mission is to identify, quantify, and rank the **most reliable T20 players under pressure** across Indian Premier League (IPL) and Men's T20 Internationals (T20Is) between 2022 and 2026.

Unlike simplistic vanity statistics (raw strike rate, nominal average, or economy rate), our framework formulates a context-aware metric: **Win Probability Added Expectation (WPAE)**. It isolates how players perform when the stakes are highest—evaluating impact relative to situation-specific peer group baselines, match leverage, collapse recovery multipliers, opponent quality, and recency-weighted decay.

```
Raw Delivery Data ──► Feature Engineering ──► Dual Win Probability Engines (LR / GBM)
                                                             │
                                                             ▼
Empirical Bayes Shrinkage ◄── Recency Decay & Elo ◄── Context WPAE & Pressure Scenarios (A–I)
          │
          ▼
300x Bootstrap CIs ──► Locked Unified, Batter, & Bowler Rankings
```

---

## 🏆 Final Locked Rankings

### 1. Unified Top 10 Players Under Pressure
*Synthesizes both batting pressure impact and bowling pressure impact, with 95% bootstrap confidence intervals and $P(\text{Top-10})$ stability.*

| Rank | Player | Player ID | Primary Role | Unified Score | Batting Score | Bowling Score | 95% Bootstrap CI | $P(\text{Top-10})$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Nathan Ellis** | `7920` | Bowler | **+0.115** | 0.000 | +0.115 | `[+0.036, +0.205]` | 0.65 |
| **2** | **Corbin Bosch** | `3629` | Bowler | **+0.109** | 0.000 | +0.109 | `[+0.025, +0.197]` | 0.58 |
| **3** | **Kagiso Rabada** | `3628` | Bowler | **+0.105** | 0.000 | +0.105 | `[+0.036, +0.181]` | 0.59 |
| **4** | **Krunal Pandya** | `4219` | All-Rounder | **+0.090** | +0.025 | +0.066 | `[+0.003, +0.175]` | 0.40 |
| **5** | **Jamie Overton** | `2382` | Bowler | **+0.084** | 0.000 | +0.084 | `[-0.003, +0.183]` | 0.35 |
| **6** | **Wanindu Hasaranga** | `4900` | All-Rounder | **+0.083** | -0.029 | +0.112 | `[-0.032, +0.209]` | 0.36 |
| **7** | **Jasprit Bumrah** | `2933` | Bowler | **+0.076** | 0.000 | +0.076 | `[+0.023, +0.129]` | 0.15 |
| **8** | **Sunil Narine** | `781` | All-Rounder | **+0.075** | -0.012 | +0.087 | `[-0.014, +0.174]` | 0.30 |
| **9** | **Ottneil Baartman** | `8805` | Bowler | **+0.074** | 0.000 | +0.074 | `[+0.000, +0.159]` | 0.28 |
| **10** | **Kuldeep Yadav** | `3644` | Bowler | **+0.073** | 0.000 | +0.073 | `[+0.012, +0.141]` | 0.25 |

---

### 2. Top 10 Batters Under Pressure (`top10_batters_unstop.csv`)
*Eligibility Filter: Full-Member / IPL player + $\ge 15$ innings + $\ge 100$ recency-weighted balls + $\ge 85$ match-situation pressure balls overall.*

| Rank | Batter | Player ID | Final Pressure Score | Total Pressure Balls | Best Phase ($\ge 85$ Balls) | Phase WPAE | $P(\text{Top-10})$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Abhishek Sharma** | `6281` | **+0.059** | 127 | Powerplay | +1.568 | 0.44 |
| **2** | **Phil Salt** | `5553` | **+0.056** | 127 | Powerplay | +0.971 | 0.41 |
| **3** | **Mitchell Marsh** | `818` | **+0.052** | 115 | Powerplay | +0.768 | 0.35 |
| **4** | **Priyansh Arya** | `25669` | **+0.050** | 95 | Powerplay | +1.335 | 0.36 |
| **5** | **Ryan Rickelton** | `4007` | **+0.048** | 126 | Powerplay | +2.256 | 0.35 |
| **6** | **Ashutosh Sharma** | `8101` | **+0.046** | 150 | Death Overs | +1.439 | 0.32 |
| **7** | **Sai Sudharsan** | `23450` | **+0.045** | 132 | Powerplay | +1.056 | 0.25 |
| **8** | **Washington Sundar** | `4823` | **+0.042** | 184 | Death Overs | +0.861 | 0.33 |
| **9** | **Josh Inglis** | `4736` | **+0.031** | 132 | Middle Overs | +0.625 | 0.16 |
| **10** | **Virat Kohli** | `47` | **+0.029** | 173 | Powerplay | +0.742 | 0.10 |

#### Phase-Specific Batter Specialists ($\ge 85$ Pressure Balls in Phase)
- **Powerplay (Overs 1–6):** 1. Ryan Rickelton (`+2.256`), 2. Abhishek Sharma (`+1.568`), 3. Priyansh Arya (`+1.335`), 4. Sai Sudharsan (`+1.056`), 5. Aiden Markram (`+1.029`).
- **Middle Overs (Overs 7–15):** 1. Reeza Hendricks (`+1.845`), 2. Rohit Sharma (`+1.677`), 3. Venkatesh Iyer (`+0.934`), 4. KL Rahul (`+0.828`), 5. Krunal Pandya (`+0.763`).
- **Death Overs (Overs 16–20):** 1. Glenn Maxwell (`+2.891`), 2. Ashutosh Sharma (`+1.439`), 3. Jason Holder (`+1.433`), 4. Mahedi Hasan (`+1.344`), 5. Nurul Hasan (`+1.278`), 8. David Miller (`+1.059`), 9. Tilak Varma (`+1.000`), 10. Rishabh Pant (`+0.952`).

---

### 3. Top 10 Bowlers Under Pressure (`top10_bowlers_unstop.csv`)
*Eligibility Filter: Full-Member / IPL player + $\ge 15$ spells + $\ge 100$ recency-weighted balls + $\ge 60$ high-leverage balls.*

| Rank | Bowler | Player ID | Final Pressure Score | Pressure Balls | Career Balls | Matches | $P(\text{Top-10})$ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Nathan Ellis** | `7920` | **+0.115** | 237 | 787 | 35 | 0.65 |
| **2** | **Wanindu Hasaranga** | `4900` | **+0.112** | 345 | 1,352 | 60 | 0.56 |
| **3** | **Corbin Bosch** | `3629` | **+0.109** | 112 | 355 | 18 | 0.57 |
| **4** | **Kagiso Rabada** | `3628` | **+0.105** | 228 | 1,073 | 49 | 0.56 |
| **5** | **Sunil Narine** | `781` | **+0.087** | 249 | 1,419 | 62 | 0.38 |
| **6** | **Jamie Overton** | `2382` | **+0.084** | 112 | 417 | 25 | 0.35 |
| **7** | **Jasprit Bumrah** | `2933` | **+0.076** | 369 | 1,683 | 75 | 0.17 |
| **8** | **Ottneil Baartman** | `8805` | **+0.074** | 95 | 380 | 18 | 0.25 |
| **9** | **Kuldeep Yadav** | `3644` | **+0.073** | 253 | 1,562 | 72 | 0.26 |
| **10** | **Liam Dawson** | `722` | **+0.073** | 69 | 428 | 20 | 0.26 |

---

## 🔬 Methodology & Quantitative Architecture

### 1. Rigorous Data Governance & Exclusions
- **Total Valid Corpus:** 807 completed matches (362 IPL + 445 Men's T20Is) encompassing 190,856 total deliveries (182,926 legal non-super-over balls).
- **Pakistan / Pakistan A Strict Exclusion:** In strict accordance with tournament stipulations, all 68 `player_id`s mapped to Pakistan (`team_id` 25) or Pakistan A (`team_id` 341) are excluded from all player ranking pools (`outputs/results/excluded_pakistan_players.csv`). However, their deliveries are retained within the match training corpus to prevent distortion of Win Probability, peer-group baselines, or Elo ratings.
- **Associate-Only Exclusion:** Players with appearances solely for Associate nations (with 0 IPL or Full-Member appearances) are excluded from the ranking consideration pool.

### 2. Dual Win Probability (WP) Engine
- **Innings 1:** Logistic Regression calibrated on balls remaining, wickets in hand, cumulative runs, venue par score, and historical phase scoring trends.
- **Innings 2:** Gradient Boosted Decision Trees (LightGBM / GBM) calibrated on required run rate (RRR), balls remaining, wickets in hand, current run rate differential, and historical chase success surfaces.
- **Leverage Index (LI):** Measures the swing potential of each delivery:
  $$\text{LI} = \frac{\Delta \text{WP}_{\text{possible}}}{\sigma(\Delta \text{WP})}$$

### 3. Nine Specific Pressure Scenarios (A through I)
Our engine tags every delivery across 9 distinct pressure archetypes:
1. **Scenario A:** Chasing under pressure ($\text{RRR} > 10.0$ or $\text{LI} > 1.5$)
2. **Scenario B:** Setting a target under pressure ($\ge 3$ wickets lost in Powerplay)
3. **Scenario C:** Knockouts & playoff matches
4. **Scenario D:** High-leverage key moments ($\text{LI} \ge 2.0$)
5. **Scenario E:** Big ask chases ($\text{Target} \ge 200$)
6. **Scenario F:** Collapse and rebuild (rescue multiplier $1.3\times$ applied)
7. **Scenario G:** High-scoring pitch environments (venue par $> 190$)
8. **Scenario H:** Pivotal match turning points ($|\Delta \text{WP}| \ge 15\%$)
9. **Scenario I:** Set-batter dismissals and bounce-back overs

### 4. Recency Weighting & Shrinkage Estimation
- **Exponential Recency Decay:** Half-life of 12 months ($T_{1/2} = 365.25 \text{ days}$):
  $$w_i = \exp\left(-\frac{\ln(2)}{365.25} \times (t_{\text{ref}} - t_i)\right)$$
- **Empirical Bayes Shrinkage:** Mitigates small sample variance by regressing small-sample performers towards population means:
  $$\text{Score} = \frac{\sum w_i \cdot \text{WPAE}_i}{\sum w_i} \cdot \frac{n_{\text{effective}}}{n_{\text{effective}} + C}$$
  where $C_{bat}$ and $C_{bowl}$ are empirical shrinkage constants derived from out-of-time cross-validation.

---

## 📁 Repository Structure

```
├── README.md                                  # Complete project documentation & guide
├── .gitignore                                 # Strict confidentiality exclusions
├── requirements.txt                           # Python package dependencies
├── SupeRR_Selector_Round1_Analysis.ipynb       # Main interactive analysis notebook
│
└── outputs/
    ├── final_submission.txt                   # Unstop competition submission copy-paste block
    ├── pipeline/                              # Sequential data science pipeline scripts
    │   ├── s0_s1_scope_and_clean.py           # Scope filtering & data dictionary audit
    │   ├── s2_pressure.py                     # Win Probability & Leverage Index engine
    │   ├── s3_wpae.py                         # Peer group baselines & WPAE calculation
    │   ├── s4_recency.py                      # 12-month decay & phase-pressure models
    │   ├── s5_shrinkage.py                    # Empirical Bayes shrinkage & bootstrap CIs
    │   ├── s6_validation.py                   # Out-of-time & split-half validation
    │   ├── s7_report.py                       # Automated chart & report compilation
    │   ├── s9_finalize.py                     # Final verification & dossier generation
    │   ├── export_pdf.py                      # PDF report generator
    │   └── analysis.ipynb                     # Pipeline walkthrough notebook
    │
    ├── results/                               # Result tables & model metrics (CSVs & JSONs)
    │   ├── top10_unified.csv                  # Unified Top 10 combined leaderboard
    │   ├── top10_batters_unstop.csv           # Top 10 Batters under pressure
    │   ├── top10_bowlers_unstop.csv           # Top 10 Bowlers under pressure
    │   ├── batter_phase_pressure_model_85b.csv# Phase-specific pressure leaders
    │   ├── excluded_pakistan_players.csv      # Audit log of 68 excluded player IDs
    │   ├── validation_summary.json            # Model validation metrics & accuracies
    │   └── sensitivity_analysis.csv           # Half-life & parameter robustness tests
    │
    └── figures/                               # 23 publication-quality diagnostic plots
        ├── rank_stability_unified.png         # Bootstrap rank distributions
        ├── calibration_inn1.png               # Innings 1 WP calibration curve
        ├── calibration_inn2.png               # Innings 2 WP calibration curve
        ├── li_by_over.png                     # Leverage index by match over
        ├── collapse_rebuild_curves.png        # Collapse rescue dynamics
        └── usual_vs_pressure_scatter.png      # Normal vs pressure performance scatter
```

---

## 🚀 How to Reproduce

### 1. Setup Environment
```bash
git clone https://github.com/Ajaymallick1220/RR-SupeRR-selecter-Round1.git
cd RR-SupeRR-selecter-Round1
pip install -r requirements.txt
```

### 2. Execute the Pipeline
Run the modular stages sequentially:
```bash
python outputs/pipeline/s0_s1_scope_and_clean.py
python outputs/pipeline/s2_pressure.py
python outputs/pipeline/s3_wpae.py
python outputs/pipeline/s4_recency.py
python outputs/pipeline/s5_shrinkage.py
python outputs/pipeline/s6_validation.py
python outputs/pipeline/s7_report.py
python outputs/pipeline/s9_finalize.py
python outputs/pipeline/s9_finalize.py verify
```

### 3. Explore the Jupyter Notebook
Open and execute [`SupeRR_Selector_Round1_Analysis.ipynb`](SupeRR_Selector_Round1_Analysis.ipynb) to inspect interactive tables, diagnostic distributions, and visual charts directly in Jupyter or GitHub.

---

## 🔒 Confidentiality & Data Protection Notice

In strict accordance with competition guidelines and non-disclosure agreements:
- **Raw Match Datasets Excluded:** Proprietary CSVs provided by Rajasthan Royals (`ball_by_ball.csv`, `matches.csv`, `playing_eleven.csv`, `teams.csv`, `competitions.csv`, `competition_team_player_mapping.csv`) and derived multi-gigabyte parquet caches are strictly excluded via `.gitignore`.
- **Proprietary Presentation Materials Excluded:** All confidential slide deck presentations (`.pptx`, `.ppt`) and internal assessment problem statements (`.pdf`, `.docx`) are omitted from public version control.
- **Safe Deliverables:** Only reproducible Python source code, self-contained analytical Jupyter notebooks, aggregated ranking tables, and non-sensitive diagnostic plots are committed.

---

**Author:** Ajay Mallick  
**Repository:** [Ajaymallick1220/RR-SupeRR-selecter-Round1](https://github.com/Ajaymallick1220/RR-SupeRR-selecter-Round1)
