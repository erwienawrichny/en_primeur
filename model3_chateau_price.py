"""
Model 3: Château Score + Classification + Biodynamic → Price
Bordeaux Left Bank | OLS regression on log(price)
19 châteaux, 114 observations, 2019–2024
Input:  château score, vintage score, classification tier, biodynamic (0/1)
Output: predicted bottle price (USD ex-tax)

Results:
  Train R²: 0.782  |  CV R²: 0.716  |  MAE: ~25%
  Score:      +1pt → +30% price  (p<0.001 ***)
  Biodynamic:      → +28% price  (p=0.045 *)
  Classification: NOT significant once score is known

Key finding: Palmer 2022 predicted ~$129, actual $435 → $306 prestige gap.
  Evidence: +1.68pt residual in Model 2 (17yr IEEE data), Liv-ex 2009
  reclassification as 1st Growth equivalent, avg $363 vs 3G peer avg $65–101.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

master = pd.read_csv('bordeaux_master_dataset.csv')

vintage_score_map = {
    2019:96.0, 2020:94.0, 2021:88.0,
    2022:97.0, 2023:93.0, 2024:91.0,
}

df = master[(master['year']>=2019) & master['price_usd'].notna()].copy()
df['log_price']     = np.log(df['price_usd'])
df['vintage_score'] = df['year'].map(vintage_score_map)
df['biodynamic']    = df['biodynamic'].fillna(0)
df['class_2'] = (df['classification']==2).astype(int)
df['class_4'] = (df['classification']==4).astype(int)
df['class_5'] = (df['classification']==5).astype(int)
df = df.dropna(subset=['vintage_score'])

# Note: class_1 dropped — no 1st Growth price data in Wine-Searcher scrape
features = ['score','vintage_score','class_2','class_4','class_5','biodynamic']
X = sm.add_constant(df[features])
y = df['log_price']
model = sm.OLS(y, X).fit()

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), df[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae_pct = (np.exp(mean_absolute_error(y, model.fittedvalues))-1)*100

print("MODEL 3: CHÂTEAU SCORE → PRICE")
print(f"N={len(df)} | Train R²={model.rsquared:.3f} | CV R²={cv_r2:.3f} | MAE={mae_pct:.1f}%")
print("\nCoefficients (3rd Growth = reference, log price):")
for v in model.params.index:
    sig = '***' if model.pvalues[v]<0.001 else '**' if model.pvalues[v]<0.01 else '*' if model.pvalues[v]<0.05 else 'ns'
    pct = (np.exp(model.params[v])-1)*100
    if v == 'const':
        print(f"  {v:<18} {model.params[v]:+.3f}  {sig}")
    else:
        print(f"  {v:<18} {model.params[v]:+.3f}  {sig}  → {pct:+.0f}% price")

def predict_price(score, vintage_score, classification, biodynamic=0, alpha=0.05):
    """Predict bottle price (USD) from château score + classification + biodynamic."""
    x = pd.DataFrame({'const':1, 'score':[score], 'vintage_score':[vintage_score],
                       'class_2':[1 if classification==2 else 0],
                       'class_4':[1 if classification==4 else 0],
                       'class_5':[1 if classification==5 else 0],
                       'biodynamic':[biodynamic]})
    pred_log = model.predict(x)[0]
    ci       = model.get_prediction(x).summary_frame(alpha=alpha)
    return {'prediction_usd': round(np.exp(pred_log),0),
            'ci_lower_usd':   round(np.exp(ci['obs_ci_lower'].values[0]),0),
            'ci_upper_usd':   round(np.exp(ci['obs_ci_upper'].values[0]),0)}
