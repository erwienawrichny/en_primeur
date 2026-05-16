"""
Model 2: Vintage Score + Classification → Château Score
Bordeaux Left Bank | 20 châteaux, 2000–2024
Requires: bordeaux_master_dataset.csv (output of data pipeline)
Input:  vintage score (from Model 1), classification (1–5), biodynamic (0/1)
Output: predicted château score (0-100)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ── Load master dataset ───────────────────────────────────────────────────────
# Run data pipeline first to generate bordeaux_master_dataset.csv
master = pd.read_csv('bordeaux_master_dataset.csv')

# ── Vintage scores (Model 1 fitted values + published consensus 2017-2024) ───
vintage_scores = {
    2000: 96.2, 2001: 91.8, 2002: 88.4, 2003: 91.5, 2004: 89.6,
    2005: 95.1, 2006: 90.2, 2007: 87.3, 2008: 89.8, 2009: 95.8,
    2010: 96.4, 2011: 90.5, 2012: 89.2, 2013: 87.6, 2014: 90.8,
    2015: 93.7, 2016: 94.9,
    # 2017–2024: published Left Bank consensus (Decanter/WA)
    2017: 90.0, 2018: 96.0, 2019: 96.0, 2020: 94.0,
    2021: 88.0, 2022: 97.0, 2023: 93.0, 2024: 91.0,
}

df = master.copy()
df['vintage_score'] = df['year'].map(vintage_scores)
df = df.dropna(subset=['vintage_score', 'score'])
df['biodynamic'] = df['biodynamic'].fillna(0)

# Classification dummies — 3rd Growth is reference category
df['class_1'] = (df['classification'] == 1).astype(int)
df['class_2'] = (df['classification'] == 2).astype(int)
df['class_4'] = (df['classification'] == 4).astype(int)
df['class_5'] = (df['classification'] == 5).astype(int)

# ── Train model ───────────────────────────────────────────────────────────────
features = ['vintage_score', 'class_1', 'class_2', 'class_4', 'class_5', 'biodynamic']
X = sm.add_constant(df[features])
y = df['score']
model = sm.OLS(y, X).fit()

# ── Evaluation ────────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), df[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae = mean_absolute_error(y, model.fittedvalues)

print("=" * 55)
print("MODEL 2: VINTAGE SCORE → CHÂTEAU SCORE")
print("=" * 55)
print(f"N = {len(df)} obs | {df['chateau'].nunique()} châteaux | 2000–2024")
print(f"Train R²:    {model.rsquared:.3f}")
print(f"Adj R²:      {model.rsquared_adj:.3f}")
print(f"5-Fold CV R²:{cv_r2:.3f}")
print(f"MAE:         {mae:.2f} pts")
print(f"\nCoefficients (3rd Growth = reference):")
for var in model.params.index:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig  = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"  {var:<25} {coef:+.3f}  (p={pval:.4f} {sig})")

# ── Château residuals (over/underperformers) ──────────────────────────────────
df['predicted'] = model.fittedvalues
df['residual']  = df['score'] - df['predicted']
residuals = (df.groupby(['chateau', 'classification'])['residual']
               .mean()
               .reset_index()
               .sort_values('residual', ascending=False))

print(f"\nChâteau residuals (avg score vs prediction):")
print(f"Positive = scores above what weather + classification predicts")
for _, r in residuals.iterrows():
    flag = " ⭐" if r['residual'] > 1.0 else " 👇" if r['residual'] < -1.0 else ""
    print(f"  {r['chateau']:<25} ({int(r['classification'])}G)  {r['residual']:+.2f}{flag}")

# ── Predict a new château/vintage ─────────────────────────────────────────────
def predict_chateau_score(vintage_score, classification, biodynamic=0, alpha=0.05):
    """
    Predict château score from vintage score + classification.

    Parameters
    ----------
    vintage_score  : float — from Model 1 prediction
    classification : int   — 1855 Classification tier (1–5)
    biodynamic     : int   — 1 if certified biodynamic, else 0
    alpha          : float — confidence level for prediction interval

    Returns
    -------
    dict with 'prediction', 'ci_lower', 'ci_upper'
    """
    x_new = pd.DataFrame({
        'const':         1,
        'vintage_score': [vintage_score],
        'class_1':       [1 if classification == 1 else 0],
        'class_2':       [1 if classification == 2 else 0],
        'class_4':       [1 if classification == 4 else 0],
        'class_5':       [1 if classification == 5 else 0],
        'biodynamic':    [biodynamic],
    })
    pred = model.predict(x_new)[0]
    ci   = model.get_prediction(x_new).summary_frame(alpha=alpha)
    return {
        'prediction': round(pred, 1),
        'ci_lower':   round(ci['obs_ci_lower'].values[0], 1),
        'ci_upper':   round(ci['obs_ci_upper'].values[0], 1),
    }

# ── 2025 predictions ──────────────────────────────────────────────────────────
vintage_2025 = 95.9  # from Model 1

print(f"\n2025 Château Score Predictions (vintage score = {vintage_2025}):")
chateau_meta = master[['chateau','classification','biodynamic']].drop_duplicates('chateau')
chateau_meta['biodynamic'] = chateau_meta['biodynamic'].fillna(0)

results = []
for _, r in chateau_meta.iterrows():
    pred = predict_chateau_score(vintage_2025, r['classification'], int(r['biodynamic']))
    results.append({**{'chateau': r['chateau'], 'classification': r['classification'],
                        'biodynamic': r['biodynamic']}, **pred})

results_df = pd.DataFrame(results).sort_values('prediction', ascending=False)
print(f"{'Château':<25} {'Cls':>3}  {'Bio':>3}  {'Pred':>5}  {'95% CI'}")
print("-" * 52)
for _, r in results_df.iterrows():
    bio = '✓' if r['biodynamic'] else ' '
    print(f"{r['chateau']:<25} {int(r['classification']):>2}G  {bio:>3}  "
          f"{r['prediction']:>5.1f}  {r['ci_lower']}–{r['ci_upper']}")

print(f"\nNote: Model predicts classification-tier averages.")
print(f"Palmer residual = +1.08pts above 3rd Growth prediction (prestige gap).")
