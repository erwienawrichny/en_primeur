"""
Model 1: Weather → Left Bank Vintage Score
Bordeaux Left Bank | Ashenfelter-style OLS regression
43 vintages 1982–2024
Input:  growing season temp (°C), harvest rainfall (mm), winter rainfall (mm)
Output: predicted Left Bank vintage score (0–100)

Results:
  Train R²: 0.725  |  CV R²: 0.488  |  MAE: 2.17 pts
  2025 forecast: 95.9 pts (95% CI: 89.3–102.4)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

vintages = pd.DataFrame([
    (1982,17.5,20,290,96),(1983,17.8,35,310,93),(1984,16.2,80,280,82),
    (1985,17.1,25,260,90),(1986,16.8,60,320,89),(1987,16.0,95,350,78),
    (1988,17.3,30,295,90),(1989,18.4,15,270,95),(1990,18.6,10,280,96),
    (1991,16.5,70,300,76),(1992,16.1,90,330,75),(1993,16.3,85,315,78),
    (1994,17.0,45,285,85),(1995,17.6,20,275,92),(1996,17.2,30,400,91),
    (1997,17.4,55,260,86),(1998,17.1,50,290,87),(1999,16.9,75,310,84),
    (2000,17.8,25,305,96),(2001,17.3,50,285,92),(2002,16.7,80,295,88),
    (2003,19.8,15,260,92),(2004,16.9,60,280,90),(2005,18.1,18,310,95),
    (2006,17.5,55,275,90),(2007,16.8,90,285,87),(2008,17.0,45,295,90),
    (2009,18.5,12,270,96),(2010,18.3,20,420,96),(2011,17.6,40,265,91),
    (2012,17.1,65,285,89),(2013,16.5,85,300,88),(2014,17.2,35,290,91),
    (2015,18.0,22,310,94),(2016,17.7,18,350,95),(2017,17.3,50,280,90),
    (2018,18.8,15,295,96),(2019,18.5,20,305,96),(2020,18.2,30,285,94),
    (2021,16.9,80,310,88),(2022,19.2,12,295,97),(2023,17.8,40,280,93),
    (2024,17.4,60,265,91),
], columns=['year','temp','harvest_rain','winter_rain','vintage_score'])

features = ['temp','harvest_rain','winter_rain']
X = sm.add_constant(vintages[features])
y = vintages['vintage_score']
model = sm.OLS(y, X).fit()

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), vintages[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae   = mean_absolute_error(y, model.fittedvalues)

print("MODEL 1: WEATHER → LEFT BANK VINTAGE SCORE")
print(f"N={len(vintages)} | Train R²={model.rsquared:.3f} | CV R²={cv_r2:.3f} | MAE={mae:.2f}pts")
print("\nCoefficients:")
for v in model.params.index:
    sig = '***' if model.pvalues[v]<0.001 else '**' if model.pvalues[v]<0.01 else '*' if model.pvalues[v]<0.05 else 'ns'
    print(f"  {v:<18} {model.params[v]:+.3f}  {sig}")

def predict_vintage(temp, harvest_rain, winter_rain, alpha=0.05):
    """Predict Left Bank vintage score from weather inputs."""
    x = pd.DataFrame({'const':1,'temp':[temp],
                       'harvest_rain':[harvest_rain],'winter_rain':[winter_rain]})
    pred = model.predict(x)[0]
    ci   = model.get_prediction(x).summary_frame(alpha=alpha)
    return {'prediction': round(pred,1),
            'ci_lower':   round(ci['obs_ci_lower'].values[0],1),
            'ci_upper':   round(ci['obs_ci_upper'].values[0],1)}

# 2025 forecast — Meteo France: 18.9°C, 22mm harvest rain, ~290mm winter rain
result = predict_vintage(18.9, 22.0, 290.0)
print(f"\n2025 forecast: {result['prediction']}pts (95% CI: {result['ci_lower']}–{result['ci_upper']})")
print("Note: winter_rain estimated. Update with actual Oct–Mar 2024/25 data.")
