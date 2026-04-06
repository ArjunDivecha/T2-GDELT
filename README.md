# T2 & GDELT factor timing

**Last updated:** 2026-04-03  
**README version:** 4.0

This repository runs **three related pipelines** that share the same economic idea (rank countries on factors, form portfolios, optimize factor weights, map to country weights, measure returns) but differ in **where factors come from** and **which filenames** they use.

| Track | Factor source | Typical prefix / key files |
|--------|----------------|----------------------------|
| **Classic T2** | `T2 Master.xlsx` (Bloomberg-style macro, valuation, momentum, etc.) | `T2_*.xlsx`, `Normalized_T2_*`, `T60.xlsx` |
| **Pure GDELT** | `GDELT.xlsx` → tidy long CSV + GDELT categories | `GDELT_*.xlsx`, `GDELT_Factors_MasterCSV.csv` |
| **T2 + GDELT combined** | Merged T2 + GDELT factor set (`Combined_T2_GDELT_Factors_MasterCSV.csv`) | `T2_GDELT_Combined_*` |

**Country returns and benchmarks** still come from the T2 side (`T2 Master.xlsx`, `Portfolio_Data.xlsx`, normalized CSV) unless a script docstring says otherwise. GDELT supplies **signals**, not replacement equity returns.

Shared configuration for GDELT and Combined **post–Step Five** paths (filenames for Steps 6–14) lives in `gdelt_track_config.py` (`GDELT_POST5`, `COMBINED_POST5`). Thin Step Six–style entry scripts import that module and call shared helpers (e.g. `gdelt_pipeline_step06_country_alphas.py`).

For **exact inputs, outputs, and versions**, read the header block at the top of each script.

---

## Prerequisites

- **Python** 3.10+ recommended  
- **Install:**

```bash
pip install -r requirements.txt
```

- Run scripts from the **repository root** (same folder as `T2 Master.xlsx` / `GDELT.xlsx` unless a script sets its own paths).
- **Excel inputs:** `.xlsx` with the sheets and columns each step expects (see per-script docs).

---

## Classic T2 pipeline (main flow)

End-to-end: build master data → normalize → benchmarks → Top 20% portfolios → monthly factor returns → rolling optimizer → country alphas → factor visuals → country weights → portfolio returns → reports → optional target optimization and regime work.

```bash
python "Step Zero Create P2P Scores.py"                    # optional P2P scores
python "Step One Create T2Master.py"
python "Step Two Create Normalized Tidy.py"
python "Step Two Point Five Create Benchmark Rets.py"
python "Step Three Top20 Portfolios Fast.py"
python "Step Four Create Monthly Top20 Returns FAST.py"
python "Step Five FAST.py"
python "Step Six Create Country alphas from Factor alphas.py"
python "Step Seven Visualize Factor Weights.py"
python "Step Eight Write Country Weights.py"
python "Step Nine Calculate Portfolio Returns.py"
python "Step Ten Create Final Report.py"
python "Step Fourteen Target Optimization.py"
```

**Often used after that (same T2 data):**

```bash
python "Step Fifteen Market Regime Analysis.py"
python "Step Sixteen Market Regime Analysis.py"
python "Step Seventeen Market Regime Analysis.py"
python "Step Eighteen Asset Class Charts.py"
python "Step Twenty PORCH.py"
python "Step FINALFINAL.py"
```

**Long–short behavior** (net/gross constraints, negative country weights where applicable) is implemented in the current Step Five FAST, Step Eight, Step Nine, and Step Fourteen stack; see those scripts for defaults.

**Archive:** older runners, grid-search notebooks, and superseded Step Three/Four/Five variants live under `Archive/`. They are **not** required for the main flow.

---

## Pure GDELT pipeline

Factors are built from **`GDELT.xlsx`**. The workbook may include documentation sheets (**`README`**, **`README_VARIABLES`**); tidying scripts **skip** those and only process **wide monthly panels** (dates in the first column).

Typical order:

```bash
python "Step Two GDELT Create Tidy.py"
python "Step Three GDELT Top20 Portfolios Fast.py"
python "Step Four GDELT Create Monthly Top20 Returns FAST.py"
python "Step Five GDELT FAST.py"
python "Step Six GDELT Create Country Alphas from Factor alphas.py"
# or equivalent wrapper:
python "Step Six Pure GDELT Create Country Alphas from Factor alphas.py"
python "Step Seven GDELT Visualize Factor Weights.py"
python "Step Eight GDELT Write Country Weights.py"
python "Step Nine GDELT Calculate Portfolio Returns.py"
python "Step Ten GDELT Create Final Report.py"
python "Step Eleven GDELT Compare Strategies.py"
python "Step Twelve GDELT MultiPeriod Factor Summary.py"
python "Step Fourteen GDELT Target Optimization.py"
```

**Supporting / optional:**

- `Step Thirteen GDELT Country Alphas Alias.py` — alias paths if you use alternate naming.  
- `Step Six Grid Search Pure GDELT *NOT PART OF FLOW*.py` — console grid over factor count × lookback (`GDELT_Optimizer.xlsx`, `Step Factor Categories GDELT.xlsx`).

Key artifacts include `GDELT_Factors_MasterCSV.csv`, `Step Factor Categories GDELT.xlsx`, `GDELT_Optimizer.xlsx`, `GDELT_T60.xlsx`, `GDELT_rolling_window_weights.xlsx`, and the `GDELT_*` outputs listed in `GDELT_POST5` inside `gdelt_track_config.py`.

**Analysis window:** GDELT-wide steps clip to the project’s GDELT analysis window (see `T2_GDELT_analysis_window.py` and Step Two GDELT docs). Month-end alignment follows the same convention as the rest of the T2 stack.

---

## T2 + GDELT combined pipeline

Merged factors from **`Step Two Point One T2 GDELT Combined Create Tidy.py`** → `Combined_T2_GDELT_Factors_MasterCSV.csv` and **`Step Factor Categories T2 GDELT Combined.xlsx`**.

**Top 20 countries (default combined path):**

```bash
python "Step Two Point One T2 GDELT Combined Create Tidy.py"
python "Step Three T2 GDELT Combined Top20 Portfolios Fast.py"
python "Step Four T2 GDELT Combined Create Monthly Top20 Returns FAST.py"
python "Step Five T2 GDELT Combined FAST.py"
python "Step Six T2 GDELT Combined Create Country Alphas from Factor alphas.py"
python "Step Seven T2 GDELT Combined Visualize Factor Weights.py"
python "Step Eight T2 GDELT Combined Write Country Weights.py"
python "Step Nine T2 GDELT Combined Calculate Portfolio Returns.py"
python "Step Ten T2 GDELT Combined Create Final Report.py"
python "Step Eleven T2 GDELT Combined Compare Strategies.py"
python "Step Twelve T2 GDELT Combined MultiPeriod Factor Summary.py"
python "Step Fourteen T2 GDELT Combined Target Optimization.py"
```

**Top 3 countries (parallel branch):** use `Step Three T2 GDELT Combined Top3 Portfolios Fast.py`, `Step Four T2 GDELT Combined Top3 Returns FAST.py`, and `Step Five T2 GDELT Combined Top3 FAST.py` instead of the Top20 Step Three/Four/Five combined scripts. Outputs use the `T2_GDELT_Combined_Top3_*` filenames (see those scripts).

**Important (combined optimizer):** `T2_GDELT_Combined_Optimizer.xlsx` can contain **many columns** that are not tradable “factors” (e.g. cumulative return helpers like `3MRet`, `6MRet`, …). **`Step Five T2 GDELT Combined FAST.py` must restrict columns to names listed in `Step Factor Categories T2 GDELT Combined.xlsx`.** Columns not in that workbook can get a loose default max weight and **inflate** backtest results. See `AGENTS.md` and the Step Five source for `USE_COVARIANCE` / `LAMBDA` behavior.

**Optional:** `Step Six Grid Search T2 GDELT Combined *NOT PART OF FLOW*.py` — same grid logic as the classic T2 grid search, pointed at combined optimizer + combined categories.

---

## Classic T2 grid search and experiments (not main flow)

- `Step Six Grid Search *NOT PART OF FLOW*.py` — T2 optimizer + `Step Factor Categories.xlsx`.  
- `Step Six Decile Analysis.py`, `Step Six SuperPower Weighting.py`, `Step Five Top Three.py`, notebooks under `Archive/` — exploratory; read each file’s header before running.

---

## Missing data and quality

Policies vary by step (forward-fill, winsorization, cross-sectional imputation for exposures, etc.). Several scripts write **data-quality sheets or logs** (for example country alpha workbooks and asset-class missing-data logs). Prefer those outputs over assumptions when auditing a run.

---

## Troubleshooting

- **Missing file:** run upstream steps; check the failing script’s **INPUT FILES** in its docstring.  
- **Date alignment:** most pipelines normalize to a **monthly** index; mixed month-end vs first-of-month sources are harmonized in the GDELT tidy steps to match the T2 convention.  
- **Optimization:** use `requirements.txt` CVXPY/OSQP versions; reduce problem size or adjust penalties if solvers fail.  
- **Combined factor count:** if combined backtests look “too good,” verify optimizer columns against `Step Factor Categories T2 GDELT Combined.xlsx`.

---

## Performance

Steps are built around **pandas / NumPy** and **CVXPY** with vectorized and warm-start patterns where possible. Suitable for modern Macs with plenty of RAM.

---

## Changelog (README only)

- **4.0 (2026-04-03):** Rewrote for three-track layout (T2, Pure GDELT, Combined); current script names; `gdelt_track_config`; GDELT README sheets; combined optimizer caveat; archived vs active grid scripts.  
- **3.0 (2025-08-31):** Prior README focused on long–short T2 and Steps 0–20 detail (superseded structure above for navigation; methodology detail still lives in script docstrings).

---

## More detail

- **`AGENTS.md`** — agent/coding notes, learned workspace facts, and combined-optimizer warnings.  
- **Each `Step *.py` file** — authoritative INPUT / OUTPUT lists and version history.
