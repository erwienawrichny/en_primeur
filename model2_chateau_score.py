"""
Model 2: Left Bank Vintage Score + Classification → Château Score
Bordeaux Left Bank | OLS regression
25 châteaux, 528 observations, 2000–2024
Input:  vintage score (from Model 1), 1855 classification tier (1–5)
Output: predicted château score (0–100)

Results:
  Train R²: 0.681  |  CV R²: 0.676  |  MAE: 1.25 pts
  3rd Growth = reference category
  5th Growth not statistically significant vs 3rd Growth (p=0.47)

Key finding: biodynamic NOT included — belongs in Model 3 (price).
Residuals capture what classification cannot explain (e.g. Palmer +1.68pts).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

master = pd.read_csv('bordeaux_master_dataset.csv')

vintage_score_map = {
    2000:96.2,2001:91.8,2002:88.4,2003:91.5,2004:89.6,
    2005:95.1,2006:90.2,2007:87.3,2008:89.8,2009:95.8,
    2010:96.4,2011:90.5,2012:89.2,2013:87.6,2014:90.8,
    2015:93.7,2016:94.9,2017:90.0,2018:96.0,2019:96.0,
    2020:94.0,2021:88.0,2022:97.0,2023:93.0,2024:91.0,
}

df = master.copy()
df['vintage_score'] = df['year'].map(vintage_score_map)
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
print(f"N={len(df)} | Train R²={model.rsquared:.3f} | CV R²={cv_r2:.3f} | MAE={mae:.2f}pts")
print("\nCoefficients (3rd Growth = reference):")
for v in model.params.index:
    sig = '***' if model.pvalues[v]<0.001 else '**' if model.pvalues[v]<0.01 else '*' if model.pvalues[v]<0.05 else 'ns'
    print(f"  {v:<18} {model.params[v]:+.3f}  {sig}")

df['predicted'] = model.fittedvalues
df['residual']  = y - model.fittedvalues
residuals = (df.groupby(['chateau','classification'])['residual']
               .mean().reset_index().sort_values('residual',ascending=False))
print("\nResiduals (avg score vs prediction):")
for _,r in residuals.iterrows():
    flag = " ⭐" if r['residual']>1.0 else " 👇" if r['residual']<-1.0 else ""
    print(f"  {r['chateau']:<25} ({int(r['classification'])}G)  {r['residual']:+.2f}{flag}")

def predict_chateau_score(vintage_score, classification, alpha=0.05):
    """Predict château score from vintage score + classification tier."""
    x = pd.DataFrame({'const':1, 'vintage_score':[vintage_score],
                       'class_1':[1 if classification==1 else 0],
                       'class_2':[1 if classification==2 else 0],
                       'class_4':[1 if classification==4 else 0],
                       'class_5':[1 if classification==5 else 0]})
    pred = model.predict(x)[0]
    ci   = model.get_prediction(x).summary_frame(alpha=alpha)
    return {'prediction': round(pred,1),
            'ci_lower':   round(ci['obs_ci_lower'].values[0],1),
            'ci_upper':   round(ci['obs_ci_upper'].values[0],1)}
