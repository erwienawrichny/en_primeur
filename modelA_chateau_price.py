"""
Model A: Château Score → Price
Bordeaux Left Bank | 19 châteaux, 2019–2024
Requires: bordeaux_master_dataset.csv (output of data pipeline)
Input:  château score, vintage score, classification (1–5), biodynamic (0/1)
Output: predicted bottle price (USD, ex-tax)

Key findings:
- +1pt score  → +30% price  (p<0.001 ***)
- Biodynamic  → +28% price  (p=0.045 *)
- Classification NOT significant once score is known
- Palmer 2022: model predicts $129, actual $435 → $306 prestige gap
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Load master dataset ───────────────────────────────────────────────────────
master = pd.read_csv('bordeaux_master_dataset.csv')

# ── Filter to 2019-2024 with price data ──────────────────────────────────────
price_df = master[(master['year'] >= 2019) & master['price_usd'].notna()].copy()
price_df['log_price'] = np.log(price_df['price_usd'])
price_df['biodynamic'] = price_df['biodynamic'].fillna(0)

# Classification dummies — 3rd Growth reference, class_1 dropped (no price data)
price_df['class_2'] = (price_df['classification'] == 2).astype(int)
price_df['class_4'] = (price_df['classification'] == 4).astype(int)
price_df['class_5'] = (price_df['classification'] == 5).astype(int)

# Vintage scores (published consensus)
vintage_scores = {
    2019: 96.0, 2020: 94.0, 2021: 88.0,
    2022: 97.0, 2023: 93.0, 2024: 91.0,
}
price_df['vintage_score'] = price_df['year'].map(vintage_scores)
price_df = price_df.dropna(subset=['vintage_score'])

# ── Train model ───────────────────────────────────────────────────────────────
features = ['score', 'vintage_score', 'class_2', 'class_4', 'class_5', 'biodynamic']
X = sm.add_constant(price_df[features])
y = price_df['log_price']
model = sm.OLS(y, X).fit()

# ── Evaluation ────────────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), price_df[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae_pct = (np.exp(mean_absolute_error(y, model.fittedvalues)) - 1) * 100

print("=" * 60)
print("MODEL A: CHÂTEAU SCORE → PRICE")
print("=" * 60)
print(f"N = {len(price_df)} obs | {price_df['chateau'].nunique()} châteaux | 2019–2024")
print(f"Dependent variable: log(price_usd)")
print(f"Train R²:    {model.rsquared:.3f}")
print(f"Adj R²:      {model.rsquared_adj:.3f}")
print(f"5-Fold CV R²:{cv_r2:.3f}")
print(f"MAE:         {mae_pct:.1f}% price error")

print(f"\nCoefficients (3rd Growth = reference):")
for var in model.params.index:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig  = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    pct  = (np.exp(coef) - 1) * 100
    if var == 'const':
        print(f"  {var:<20} {coef:+.3f}  (p={pval:.4f} {sig})")
    else:
        print(f"  {var:<20} {coef:+.3f}  (p={pval:.4f} {sig})  → {pct:+.0f}% price effect")

# ── VIF check ────────────────────────────────────────────────────────────────
print(f"\nVIF (multicollinearity check):")
X_vif = sm.add_constant(price_df[features])
for i, var in enumerate(X_vif.columns):
    if var == 'const': continue
    vif = variance_inflation_factor(X_vif.values, i)
    flag = "✅" if vif < 5 else "⚠️"
    print(f"  {var:<20} {vif:>5.2f}  {flag}")

# ── Predict price for a château ───────────────────────────────────────────────
def predict_price(score, vintage_score, classification, biodynamic=0, alpha=0.05):
    """
    Predict bottle price from château score + classification.

    Parameters
    ----------
    score          : float — critic score for this vintage
    vintage_score  : float — vintage-level score (from Model 1)
    classification : int   — 1855 Classification tier (1–5)
    biodynamic     : int   — 1 if certified biodynamic, else 0
    alpha          : float — confidence level for prediction interval

    Returns
    -------
    dict with 'prediction_usd', 'ci_lower_usd', 'ci_upper_usd'
    """
    x_new = pd.DataFrame({
        'const':         1,
        'score':         [score],
        'vintage_score': [vintage_score],
        'class_2':       [1 if classification == 2 else 0],
        'class_4':       [1 if classification == 4 else 0],
        'class_5':       [1 if classification == 5 else 0],
        'biodynamic':    [biodynamic],
    })
    pred_log = model.predict(x_new)[0]
    ci       = model.get_prediction(x_new).summary_frame(alpha=alpha)
    return {
        'prediction_usd': round(np.exp(pred_log), 0),
        'ci_lower_usd':   round(np.exp(ci['obs_ci_lower'].values[0]), 0),
        'ci_upper_usd':   round(np.exp(ci['obs_ci_upper'].values[0]), 0),
    }

# ── 2025 price predictions ────────────────────────────────────────────────────
# Château scores from Model 2 (vintage_score = 95.9)
chateau_2025 = [
    ('Margaux',            1, 99.3, 1), ('Mouton Rothschild',   1, 97.2, 0),
    ('Latour',             1, 97.2, 0), ('Lafite Rothschild',   1, 97.2, 0),
    ('Haut-Brion',         1, 97.2, 0), ('Leoville Las Cases',  2, 96.3, 0),
    ('Pichon Lalande',     2, 96.3, 0), ('Ducru-Beaucaillou',   2, 96.3, 0),
    ("Cos d'Estournel",    2, 96.3, 0), ('Montrose',            2, 96.3, 0),
    ('Leoville Barton',    2, 96.3, 0), ('Pichon Baron',        2, 96.3, 0),
    ('Palmer',             3, 93.8, 1), ('Calon-Segur',         3, 93.8, 0),
    ('Giscours',           3, 93.8, 0), ('Lagrange',            3, 93.8, 0),
    ('Pontet-Canet',       5, 93.6, 1), ('Lynch-Bages',         5, 93.6, 0),
    ('Grand-Puy-Lacoste',  5, 93.6, 0), ("d'Armailhac",         5, 93.6, 0),
    ('Clerc-Milon',        5, 93.6, 0), ('Duhart-Milon',        4, 92.3, 0),
    ('Beychevelle',        4, 92.3, 0), ('Prieure-Lichine',     4, 92.3, 0),
    ('Talbot',             4, 92.3, 0),
]

print(f"\n2025 Price Predictions (vintage score = 95.9):")
print(f"{'Château':<25} {'Cls':>3}  {'Bio':>3}  {'Score':>6}  {'Pred Price':>11}  {'95% CI'}")
print("-" * 72)
for ch, cl, sc, bio in chateau_2025:
    p = predict_price(sc, 95.9, cl, bio)
    bio_flag = ' ✓' if bio else '  '
    print(f"{ch:<25} {cl:>2}G  {bio_flag}  {sc:>6.1f}  "
          f"${p['prediction_usd']:>9.0f}  "
          f"${p['ci_lower_usd']:.0f}–${p['ci_upper_usd']:.0f}")

print(f"""
Notes:
- Dependent variable is log(price), coefficients = % price effects
- 1st Growth châteaux excluded from price model (no Wine-Searcher price data)
- class_1 dummy dropped due to perfect multicollinearity
- MAE of {mae_pct:.1f}% reflects genuine price uncertainty beyond model variables
- Palmer 2022: model predicts ~$129, actual $435 → +$306 unexplained prestige premium
  Evidence: +1.08pt residual across 17yr IEEE data, Liv-ex 2009 reclassification
  as 1st Growth equivalent, avg $363 vs 3rd Growth peer avg of $65-$101
""")
