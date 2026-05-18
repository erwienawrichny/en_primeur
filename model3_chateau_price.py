"""
Model 3: Château Score + Classification + Biodynamic → Price
Bordeaux Left Bank | OLS regression on log(price)
25 châteaux, 121 observations, TRAINED on 2019–2023, PREDICTS 2024
Input:  château score, vintage score, classification (1–5), biodynamic (0/1)
Output: predicted bottle price (USD ex-tax)

Results (trained 2019-2023):
  Train R²: 0.895  |  CV R²: 0.864  |  MAE: ~23.5%
  score:      +1pt → +20% price  (p<0.001 ***)
  class_1:         → +220% vs 3rd Growth (p<0.001 ***)
  class_2:         → +35% vs 3rd Growth  (p=0.004 **)
  biodynamic:      → +36% premium        (p<0.001 ***)
  class_4/5: not significant
  vintage_score: not significant

2024 holdout validation:
  Avg absolute error: 45.4% — model systematically OVERPREDICTS 2024
  Reason: 2024 was a soft vintage in a soft market. Model trained on
  2019-2023 (including the exceptional 2022 vintage) overestimates
  what a modest vintage commands.

Palmer prestige gap (2024):
  Model predicts $203, actual $202 — almost perfect match!
  Unlike 2022 where model predicted ~$129 vs actual $435,
  2024 Palmer priced more modestly (soft market + soft vintage).

Key note: class_1 now statistically significant with full 25-château data.
Previously dropped due to no 1st Growth price data.
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

master = pd.read_csv('bordeaux_master_dataset.csv')

vintage_score_map = {
    2019:98,2020:97,2021:90,2022:99,2023:93,2024:87
}

# Train on 2019-2023
df = master[(master['year']>=2019) & (master['year']<=2023) &
            master['price_usd'].notna()].copy()
df['log_price']     = np.log(df['price_usd'])
df['vintage_score'] = df['year'].map(vintage_score_map)
df['biodynamic']    = df['biodynamic'].fillna(0)
df['class_1'] = (df['classification']==1).astype(int)
df['class_2'] = (df['classification']==2).astype(int)
df['class_4'] = (df['classification']==4).astype(int)
df['class_5'] = (df['classification']==5).astype(int)
df = df.dropna(subset=['vintage_score'])

features = ['score','vintage_score','class_1','class_2','class_4','class_5','biodynamic']
X = sm.add_constant(df[features])
y = df['log_price']
model = sm.OLS(y, X).fit()

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), df[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae_pct = (np.exp(mean_absolute_error(y, model.fittedvalues))-1)*100

print("MODEL 3: CHÂTEAU SCORE → PRICE (trained 2019-2023)")
print(f"N={len(df)} | Train R²={model.rsquared:.3f} | CV R²={cv_r2:.3f} | MAE={mae_pct:.1f}%")
print("\nCoefficients (3rd Growth = reference, dep var = log price):")
for v in model.params.index:
    sig = '***' if model.pvalues[v]<0.001 else '**' if model.pvalues[v]<0.01 \
          else '*' if model.pvalues[v]<0.05 else 'ns'
    pct = (np.exp(model.params[v])-1)*100
    if v == 'const':
        print(f"  {v:<20} {model.params[v]:+.3f}  p={model.pvalues[v]:.4f}  {sig}")
    else:
        print(f"  {v:<20} {model.params[v]:+.3f}  p={model.pvalues[v]:.4f}  {sig}  → {pct:+.0f}%")

def predict_price(score, vintage_score, classification, biodynamic=0, alpha=0.05):
    """Predict bottle price (USD) from château score + classification + biodynamic."""
    x = pd.DataFrame({'const':1,'score':[score],'vintage_score':[vintage_score],
                       'class_1':[1 if classification==1 else 0],
                       'class_2':[1 if classification==2 else 0],
                       'class_4':[1 if classification==4 else 0],
                       'class_5':[1 if classification==5 else 0],
                       'biodynamic':[biodynamic]})
    pred_log = model.predict(x)[0]
    ci       = model.get_prediction(x).summary_frame(alpha=alpha)
    return {'prediction_usd': round(np.exp(pred_log), 0),
            'ci_lower_usd':   round(np.exp(ci['obs_ci_lower'].values[0]), 0),
            'ci_upper_usd':   round(np.exp(ci['obs_ci_upper'].values[0]), 0)}

# 2024 out-of-sample validation
print("\n── 2024 Out-of-sample predictions ──")
df_2024 = master[(master['year']==2024) & master['price_usd'].notna()].copy()
df_2024['vintage_score'] = 87
df_2024['biodynamic']    = df_2024['biodynamic'].fillna(0)
df_2024['class_1']=(df_2024['classification']==1).astype(int)
df_2024['class_2']=(df_2024['classification']==2).astype(int)
df_2024['class_4']=(df_2024['classification']==4).astype(int)
df_2024['class_5']=(df_2024['classification']==5).astype(int)
X_2024 = sm.add_constant(df_2024[features], has_constant='add')
df_2024['pred_price'] = np.exp(model.predict(X_2024))
df_2024['error_pct']  = (df_2024['pred_price']-df_2024['price_usd'])/df_2024['price_usd']*100

print(f"{'Château':<25} {'Class':>4}  {'Actual':>8}  {'Predicted':>10}  {'Error%':>8}")
print("-"*62)
for _,r in df_2024.sort_values('classification').iterrows():
    print(f"{r['chateau']:<25} {int(r['classification']):>3}G  "
          f"${r['price_usd']:>7.0f}  ${r['pred_price']:>9.0f}  {r['error_pct']:>+7.1f}%")
print(f"\nAvg absolute error: {df_2024['error_pct'].abs().mean():.1f}%")
