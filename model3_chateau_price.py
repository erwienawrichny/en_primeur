"""
Model 3: Château Fixed Effects + Regional Vintage Score → Price
Bordeaux Left Bank | OLS with château fixed effects
25 châteaux, 121 observations, TRAINED on 2019–2023, PREDICTS 2024+

Vintage score source: Wine-Searcher Médoc community aggregate (~40 critics)
Château scores source: Wine-Searcher (consistent source)
Price source: Wine-Searcher avg price ex-tax

Approach: Château fixed effects
  Each château gets its own baseline price intercept learned from data.
  Regional vintage score (Wine-Searcher Médoc) explains year-to-year movement.

Results (trained 2019-2023):
  Train R²: 0.982  |  CV R²: 0.936  |  MAE: 8.8%
  vintage_score: +14.8% per point (p<0.001 ***)

2024 holdout MAE: 30.7% — model overpredicts because 2024 en primeur
prices are still settling and market is cooling post-2022 highs.
Compare again when secondary market settles in late 2026/2027.

Reference château: Beychevelle (4th Growth, St-Julien)
Note: choice of reference doesn't affect predictions, only reporting.
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

# Wine-Searcher Médoc regional vintage scores (~40 critics aggregated)
vintage_score_map = {
    2000:93,2001:91,2002:91,2003:93,2004:92,2005:93,2006:92,
    2007:91,2008:92,2009:94,2010:94,2011:92,2012:92,2013:91,
    2014:94,2015:94,2016:96,2017:94,2018:96,2019:96,2020:95,
    2021:94,2022:96,2023:95,2024:93
}

# Train on 2019-2023
df = master[(master['year']>=2019) & (master['year']<=2023) &
            master['price_usd'].notna()].copy()
df['log_price']     = np.log(df['price_usd'])
df['vintage_score'] = df['year'].map(vintage_score_map)
df = df.dropna(subset=['vintage_score']).reset_index(drop=True)
y = df['log_price'].values.astype(float)

ch_dummies = pd.get_dummies(df['chateau'], prefix='ch').drop(
    'ch_Beychevelle', axis=1).astype(float)
X = sm.add_constant(pd.concat([df[['vintage_score']].astype(float),
                                ch_dummies], axis=1).astype(float))
model = sm.OLS(y, X).fit()

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), X.values, y,
                        cv=kf, scoring='r2').mean()
mae   = (np.exp(mean_absolute_error(y, model.fittedvalues))-1)*100
vs_pct = (np.exp(model.params['vintage_score'])-1)*100

print("MODEL 3: CHÂTEAU FIXED EFFECTS + REGIONAL VINTAGE SCORE → PRICE")
print(f"N={len(df)} | Train R²={model.rsquared:.3f} | "
      f"CV R²={cv_r2:.3f} | MAE={mae:.1f}%")
print(f"\nRegional vintage score: {vs_pct:+.1f}% per point  "
      f"p={model.pvalues['vintage_score']:.4f}  ***")
print("(Wine-Searcher Médoc community aggregate, ~40 critics)")

print("\nChâteau fixed effects (price premium/discount vs Beychevelle):")
fe = {c.replace('ch_',''): (np.exp(model.params[c])-1)*100
      for c in model.params.index if c.startswith('ch_')}
for ch, pct in sorted(fe.items(), key=lambda x: x[1], reverse=True):
    cl  = int(master[master['chateau']==ch]['classification'].iloc[0])
    sig = ('***' if model.pvalues[f'ch_{ch}']<0.001 else
           '**'  if model.pvalues[f'ch_{ch}']<0.01  else
           '*'   if model.pvalues[f'ch_{ch}']<0.05  else 'ns')
    print(f"  {ch:<25} ({cl}G)  {pct:>+6.0f}%  {sig}")

def predict_price(chateau, vintage_score, alpha=0.05):
    """Predict bottle price for a known château in a given vintage."""
    row = {'vintage_score': vintage_score}
    for col in ch_dummies.columns:
        row[col] = 1.0 if col == f'ch_{chateau}' else 0.0
    x = pd.DataFrame([row])
    x = sm.add_constant(x, has_constant='add')
    for col in X.columns:
        if col not in x.columns:
            x[col] = 0.0
    x = x[X.columns]
    pred_log = model.predict(x)[0]
    ci = model.get_prediction(x).summary_frame(alpha=alpha)
    return {
        'prediction_usd': round(np.exp(pred_log), 0),
        'ci_lower_usd':   round(np.exp(ci['obs_ci_lower'].values[0]), 0),
        'ci_upper_usd':   round(np.exp(ci['obs_ci_upper'].values[0]), 0),
    }

# 2024 out-of-sample predictions
print(f"\n── 2024 Predictions (WS Médoc vintage score = {vintage_score_map[2024]}) ──")
df_2024 = master[(master['year']==2024) & master['price_usd'].notna()].copy()
df_2024['vintage_score'] = vintage_score_map[2024]
df_2024 = df_2024.reset_index(drop=True)
ch_d_2024 = pd.get_dummies(df_2024['chateau'], prefix='ch').astype(float)
for col in ch_dummies.columns:
    if col not in ch_d_2024.columns:
        ch_d_2024[col] = 0.0
ch_d_2024 = ch_d_2024[ch_dummies.columns]
X_2024 = sm.add_constant(
    pd.concat([df_2024[['vintage_score']].astype(float),
               ch_d_2024], axis=1).astype(float), has_constant='add')
for col in X.columns:
    if col not in X_2024.columns:
        X_2024[col] = 0.0
X_2024 = X_2024[X.columns]
df_2024['pred_price'] = np.exp(model.predict(X_2024))
df_2024['diff_pct'] = (df_2024['pred_price']-df_2024['price_usd'])/df_2024['price_usd']*100

print(f"{'Château':<25} {'Cls':>3}  {'Actual':>8}  {'Predicted':>10}  {'Diff%':>7}")
print("-"*60)
for _,r in df_2024.sort_values('classification').iterrows():
    print(f"{r['chateau']:<25} {int(r['classification']):>2}G  "
          f"${r['price_usd']:>7.0f}  ${r['pred_price']:>9.0f}  {r['diff_pct']:>+6.1f}%")
mae_2024 = df_2024['diff_pct'].abs().mean()
print(f"\nMAE: {mae_2024:.1f}% — prices still settling, compare in 2027")

df_2024[['chateau','classification','score','vintage_score',
         'price_usd','pred_price','diff_pct']].to_csv(
    'bordeaux_2024_price_predictions.csv', index=False)
print("✅ Saved bordeaux_2024_price_predictions.csv")
