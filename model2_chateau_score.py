"""
Model 2: Vintage Score + Classification → Château Score
Bordeaux Left Bank | OLS regression
25 châteaux, 528 observations, 2000–2024
Input:  vintage_score (WCI scores, from Model 1 framework), 1855 classification (1–5)
Output: predicted château score (0–100)

Results:
  Train R²: 0.657 (65.7%)  |  CV R²: 0.653 (65.3%)  |  MAE: 1.28 pts
  Gap: 0.4% — no overfitting

Coefficients (3rd Growth = reference):
  vintage_score: +0.330 (p<0.001 ***)
  1st Growth:    +3.447 (p<0.001 ***)
  2nd Growth:    +2.312 (p<0.001 ***)
  4th Growth:    -1.670 (p<0.001 ***)
  5th Growth:    -0.176 (p=0.44  ns) — same as 3rd Growth

Key residuals (avg score vs prediction):
  Palmer          +1.68 ⭐   Pontet-Canet    +1.50 ⭐
  Lynch-Bages     +1.10 ⭐   Leoville Las Cases +1.05 ⭐
  Lagrange        -1.32 👇   d'Armailhac     -1.62 👇

Note: Earlier model iterations using estimated 2017-2024 data produced
inflated residuals (Palmer +2.46, Pontet-Canet +2.26). These figures
should not be used. Current residuals use real data only:
  - IEEE BordeauxWines dataset (2000-2016)
  - Wine-Searcher scraped scores (2017-2024)

Biodynamic NOT included — belongs in Model 3 (price), not Model 2 (score).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

master = pd.read_csv('bordeaux_master_dataset.csv')

# WCI vintage scores (Wine Cellar Insider)
vintage_score_map = {
    2000:99,2001:92,2002:89,2003:93,2004:91,2005:98,2006:91,
    2007:87,2008:91,2009:98,2010:97,2011:87,2012:90,2013:83,
    2014:94,2015:97,2016:99,2017:92,2018:98,2019:98,2020:97,
    2021:90,2022:99,2023:93,2024:87
}

df = master.copy()
df['vintage_score'] = df['year'].map(vintage_score_map)
df['biodynamic']    = df['biodynamic'].fillna(0)
df = df.dropna(subset=['vintage_score','score'])
df['class_1'] = (df['classification']==1).astype(int)
df['class_2'] = (df['classification']==2).astype(int)
df['class_4'] = (df['classification']==4).astype(int)
df['class_5'] = (df['classification']==5).astype(int)

features = ['vintage_score','class_1','class_2','class_4','class_5']
X = sm.add_constant(df[features])
y = df['score']
model = sm.OLS(y, X).fit()

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), df[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae   = mean_absolute_error(y, model.fittedvalues)

print("MODEL 2: VINTAGE SCORE + CLASSIFICATION → CHÂTEAU SCORE")
print(f"N={len(df)} | Train R²={model.rsquared:.3f} ({model.rsquared*100:.1f}%) | "
      f"CV R²={cv_r2:.3f} ({cv_r2*100:.1f}%) | MAE={mae:.2f}pts")
print("\nCoefficients (3rd Growth = reference):")
for v in model.params.index:
    sig = '***' if model.pvalues[v]<0.001 else '**' if model.pvalues[v]<0.01 \
          else '*' if model.pvalues[v]<0.05 else 'ns'
    print(f"  {v:<20} {model.params[v]:+.3f}  p={model.pvalues[v]:.4f}  {sig}")

df['predicted'] = model.fittedvalues
df['residual']  = y - model.fittedvalues
residuals = (df.groupby(['chateau','classification'])['residual']
               .mean().reset_index().sort_values('residual', ascending=False))
print("\nResiduals (avg score vs prediction):")
for _, r in residuals.iterrows():
    flag = "⭐" if r['residual']>1 else "👇" if r['residual']<-1 else ""
    print(f"  {r['chateau']:<25} ({int(r['classification'])}G)  {r['residual']:+.2f} {flag}")

def predict_chateau_score(vintage_score, classification, alpha=0.05):
    """
    Predict château score from vintage score + classification tier.

    Parameters
    ----------
    vintage_score  : float — WCI Left Bank vintage score for that year
    classification : int   — 1855 Classification tier (1–5)
    alpha          : float — confidence level for prediction interval

    Returns
    -------
    dict with prediction, ci_lower, ci_upper
    """
    x = pd.DataFrame({'const':1,'vintage_score':[vintage_score],
                       'class_1':[1 if classification==1 else 0],
                       'class_2':[1 if classification==2 else 0],
                       'class_4':[1 if classification==4 else 0],
                       'class_5':[1 if classification==5 else 0]})
    pred = model.predict(x)[0]
    ci   = model.get_prediction(x).summary_frame(alpha=alpha)
    return {'prediction': round(pred,1),
            'ci_lower':   round(ci['obs_ci_lower'].values[0],1),
            'ci_upper':   round(ci['obs_ci_upper'].values[0],1)}
