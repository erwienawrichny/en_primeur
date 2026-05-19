"""
Model 3: Château Fixed Effects + Vintage Score → Price
Bordeaux Left Bank | OLS with château fixed effects
25 châteaux, 121 observations, TRAINED on 2019–2023, PREDICTS 2024+

Approach: Château fixed effects
  Each château gets its own baseline price (intercept) learned from data.
  Vintage score moves all châteaux up/down from their own baseline.
  This captures everything permanently true about a château (terroir,
  brand, winemaker, collector demand) without needing to explain it.

Results (trained 2019-2023):
  Train R²: 0.988  |  CV R²: 0.945  |  MAE: 7.0%
  vintage_score: +3.6% per point (p<0.001 ***)

vs Model A (classification dummies):
  Train R²: 0.895  |  CV R²: 0.864  |  MAE: 23.5%

Key insight: Once you control for château identity, vintage score alone
explains year-to-year price movement. Individual critic scores add noise.
The château's reputation is already priced in — vintage quality is the swing.

2024 holdout MAE: 29.5% — model overpredicts because 2024 is a soft
vintage in a soft post-2022 market. Lower-prestige châteaux nearly perfect.
Prices still settling — compare again in 2027.

Reference château: Beychevelle (4th Growth, St-Julien)
All château fixed effects are premiums/discounts vs Beychevelle.

Limitation: Cannot predict a new château with no price history.
Designed for predicting KNOWN châteaux in future vintages.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Load data ─────────────────────────────────────────────────────────────────
master = pd.read_csv('bordeaux_master_dataset.csv')

vintage_score_map = {
    2019:98, 2020:97, 2021:90, 2022:99, 2023:93, 2024:87
}

# ── Train on 2019-2023 ────────────────────────────────────────────────────────
df = master[(master['year']>=2019) & (master['year']<=2023) &
            master['price_usd'].notna()].copy()
df['log_price']     = np.log(df['price_usd'])
df['vintage_score'] = df['year'].map(vintage_score_map)
df = df.dropna(subset=['vintage_score']).reset_index(drop=True)
y = df['log_price'].values.astype(float)

# Château dummies — Beychevelle as reference
ch_dummies = pd.get_dummies(df['chateau'], prefix='ch', drop_first=False)
ch_dummies = ch_dummies.drop('ch_Beychevelle', axis=1).astype(float)

X = sm.add_constant(
    pd.concat([df[['vintage_score']].astype(float), ch_dummies], axis=1).astype(float))
model = sm.OLS(y, X).fit()

# ── Evaluation ────────────────────────────────────────────────────────────────
kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), X.values, y,
                        cv=kf, scoring='r2').mean()
mae   = (np.exp(mean_absolute_error(y, model.fittedvalues))-1)*100

print("MODEL 3: CHÂTEAU FIXED EFFECTS + VINTAGE SCORE → PRICE")
print(f"N={len(df)} | Train R²={model.rsquared:.3f} | "
      f"CV R²={cv_r2:.3f} | MAE={mae:.1f}%")

# ── Vintage score coefficient ─────────────────────────────────────────────────
vs_coef = model.params['vintage_score']
vs_pval = model.pvalues['vintage_score']
vs_pct  = (np.exp(vs_coef)-1)*100
print(f"\nVintage score: {vs_coef:+.3f} → {vs_pct:+.1f}% per point  "
      f"p={vs_pval:.4f}  ***")

# ── Château fixed effects ─────────────────────────────────────────────────────
print("\nChâteau fixed effects (price premium vs Beychevelle):")
fe_cols = [c for c in model.params.index if c.startswith('ch_')]
fe = pd.Series({c.replace('ch_',''): model.params[c] for c in fe_cols})
fe = fe.sort_values(ascending=False)
for ch, coef in fe.items():
    pval = model.pvalues[f'ch_{ch}']
    pct  = (np.exp(coef)-1)*100
    cl   = int(master[master['chateau']==ch]['classification'].iloc[0])
    sig  = '***' if pval<0.001 else '**' if pval<0.01 else '*' if pval<0.05 else 'ns'
    print(f"  {ch:<25} ({cl}G)  {pct:>+6.0f}%  {sig}")

# ── Predict price for a château/vintage ──────────────────────────────────────
def predict_price(chateau, vintage_score, alpha=0.05):
    """
    Predict bottle price (USD) for a known château in a given vintage.

    Parameters
    ----------
    chateau       : str   — must match a château name in training data
    vintage_score : int   — vintage quality score (e.g. 99 for 2022)
    alpha         : float — confidence level for prediction interval

    Returns
    -------
    dict with prediction_usd, ci_lower_usd, ci_upper_usd
    """
    row = {'vintage_score': vintage_score}
    for col in ch_dummies.columns:
        row[col] = 1.0 if col == f'ch_{chateau}' else 0.0
    x = sm.add_constant(pd.DataFrame([row]), has_constant='add')
    # Align columns
    for col in X.columns:
        if col not in x.columns:
            x[col] = 0.0
    x = x[X.columns]
    pred_log = model.predict(x)[0]
    ci       = model.get_prediction(x).summary_frame(alpha=alpha)
    return {
        'prediction_usd': round(np.exp(pred_log), 0),
        'ci_lower_usd':   round(np.exp(ci['obs_ci_lower'].values[0]), 0),
        'ci_upper_usd':   round(np.exp(ci['obs_ci_upper'].values[0]), 0),
    }

# ── 2024 out-of-sample validation ─────────────────────────────────────────────
print("\n── 2024 Out-of-sample predictions (vintage score = 87) ──")
df_2024 = master[(master['year']==2024) & master['price_usd'].notna()].copy()

ch_d_2024 = pd.get_dummies(df_2024['chateau'], prefix='ch', drop_first=False)
for col in ch_dummies.columns:
    if col not in ch_d_2024.columns:
        ch_d_2024[col] = 0
ch_d_2024 = ch_d_2024[ch_dummies.columns].astype(float)

df_2024['vintage_score'] = 87
X_2024 = sm.add_constant(
    pd.concat([df_2024[['vintage_score']].astype(float).reset_index(drop=True),
               ch_d_2024.reset_index(drop=True)], axis=1).astype(float),
    has_constant='add')
for col in X.columns:
    if col not in X_2024.columns:
        X_2024[col] = 0.0
X_2024 = X_2024[X.columns]

df_2024 = df_2024.reset_index(drop=True)
df_2024['pred_price'] = np.exp(model.predict(X_2024))
df_2024['diff_pct']   = (df_2024['pred_price']-df_2024['price_usd'])/df_2024['price_usd']*100

print(f"{'Château':<25} {'Class':>4}  {'Actual':>8}  {'Predicted':>10}  {'Diff%':>7}")
print("-"*60)
for _,r in df_2024.sort_values('classification').iterrows():
    print(f"{r['chateau']:<25} {int(r['classification']):>3}G  "
          f"${r['price_usd']:>7.0f}  ${r['pred_price']:>9.0f}  {r['diff_pct']:>+6.1f}%")
mae_2024 = df_2024['diff_pct'].abs().mean()
print(f"\nMAE on 2024 holdout: {mae_2024:.1f}%")
print("Note: 2024 prices still settling — compare again in 2027")

# Save 2024 predictions
df_2024[['chateau','classification','score','vintage_score',
          'price_usd','pred_price','diff_pct']].to_csv(
    'bordeaux_2024_price_predictions.csv', index=False)
print("✅ Saved bordeaux_2024_price_predictions.csv")
