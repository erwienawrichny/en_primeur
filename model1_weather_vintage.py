"""
Model 1: Weather → Left Bank Vintage Score
Bordeaux Left Bank | OLS regression
25 vintages 2000–2024 | Daily weather data from Meteostat (Bordeaux-Merignac, station LFBD)

Variables:
  tmax_apr_sep        — avg daily maximum temperature Apr–Sep (°C)
  harvest_rain_aug_sep — total precipitation Aug+Sep (mm)

Results:
  Train R²: 0.455 (45.5%)  |  CV R²: 0.130 (13.0%)  |  MAE: 2.73 pts
  tmax: +1.668 pts per °C (p=0.034*)
  harvest_rain: -0.056 pts per mm (p=0.009**)
  winter_rain: dropped — not significant in modern era (p>0.5)

Note: Ashenfelter's 1988 model achieved ~89% R² with data spanning 1952-1980s
where vintages ranged from catastrophic (50pts) to legendary (96pts). In the
modern era (2000-2024) winemaking improvements have compressed the score range
(83-99), weakening the weather signal. The relationship is real but narrower.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ── Dataset: 25 vintages 2000–2024 ───────────────────────────────────────────
# Weather: Meteostat daily data, Bordeaux-Merignac station (LFBD)
# Scores: Wine Cellar Insider vintage chart (publicly available)
# tmax = avg daily max temp Apr-Sep | harvest_rain = Aug+Sep total mm
vintages = pd.DataFrame([
    (2000, 23.9,  69, 99), (2001, 23.1,  91, 92), (2002, 22.7, 143, 89),
    (2003, 25.9,  68, 93), (2004, 23.5, 122, 91), (2005, 24.6,  71, 98),
    (2006, 24.9, 171, 91), (2007, 23.3, 121, 87), (2008, 22.9, 148, 91),
    (2009, 24.3,  72, 98), (2010, 23.9,  40, 97), (2011, 25.0, 114, 87),
    (2012, 23.6,  78, 90), (2013, 23.0, 132, 83), (2014, 23.8, 101, 94),
    (2015, 24.8, 125, 97), (2016, 24.1,  76, 99), (2017, 24.5, 102, 92),
    (2018, 25.6,  22, 98), (2019, 24.6, 102, 98), (2020, 25.3, 157, 97),
    (2021, 23.5, 105, 90), (2022, 26.6,  66, 99), (2023, 25.8, 127, 93),
    (2024, 23.8, 165, 87),
], columns=['year', 'tmax_apr_sep', 'harvest_rain_aug_sep', 'vintage_score'])

features = ['tmax_apr_sep', 'harvest_rain_aug_sep']
X = sm.add_constant(vintages[features])
y = vintages['vintage_score']
model = sm.OLS(y, X).fit()

kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), vintages[features].values,
                        y.values, cv=kf, scoring='r2').mean()
mae   = mean_absolute_error(y, model.fittedvalues)

print("MODEL 1: WEATHER → LEFT BANK VINTAGE SCORE")
print(f"N={len(vintages)} | Train R²={model.rsquared:.3f} ({model.rsquared*100:.1f}%) | "
      f"CV R²={cv_r2:.3f} ({cv_r2*100:.1f}%) | MAE={mae:.2f}pts")
print("\nCoefficients:")
for v in model.params.index:
    sig = '***' if model.pvalues[v]<0.001 else '**' if model.pvalues[v]<0.01 \
          else '*' if model.pvalues[v]<0.05 else 'ns'
    print(f"  {v:<25} {model.params[v]:+.3f}  p={model.pvalues[v]:.4f}  {sig}")

def predict_vintage(tmax_apr_sep, harvest_rain_aug_sep, alpha=0.05):
    """Predict vintage score from weather. Returns prediction + 95% CI."""
    x = pd.DataFrame({'const': 1, 'tmax_apr_sep': [tmax_apr_sep],
                       'harvest_rain_aug_sep': [harvest_rain_aug_sep]})
    pred = model.predict(x)[0]
    ci   = model.get_prediction(x).summary_frame(alpha=alpha)
    return {'prediction': round(pred, 1),
            'ci_lower':   round(ci['obs_ci_lower'].values[0], 1),
            'ci_upper':   round(ci['obs_ci_upper'].values[0], 1)}

# 2025 forecast — hot dry summer, Meteostat daily data
result = predict_vintage(tmax_apr_sep=27.5, harvest_rain_aug_sep=18.0)
print(f"\n2025 forecast: {result['prediction']}pts (95% CI: {result['ci_lower']}–{result['ci_upper']})")
print("Note: tmax and harvest_rain estimated from early season reports.")
