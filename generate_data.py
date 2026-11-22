"""
generate_data.py
Creates the synthetic residential housing dataset (housing.csv) used in the
House Price Prediction project (26MCAC103 - Business Statistics Using Python).

Fixed seed (numpy default_rng(42)) -> the file is reproduced exactly on every run.
Usage:  python generate_data.py
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 1500
zones = {'Central CBD':(9.0,'Urban',4.6,3.0),'IT Corridor':(5.0,'Urban',4.1,5.0),
         'West Enclave':(2.0,'Urban',3.8,7.0),'Metro Suburb':(-2.0,'Suburban',3.3,11.0),
         'East Township':(-5.0,'Suburban',2.8,15.0),'Outer Ring':(-8.0,'Peripheral',2.2,22.0)}
zn = rng.choice(list(zones), N, p=[.14,.18,.16,.20,.17,.15])
area = np.clip(rng.normal(1440, 300, N), 600, 2390).round(-1)
bed = np.clip(np.round((area-450)/330 + rng.normal(0,.55,N)),1,5).astype(int)
bath = np.clip(bed - rng.integers(0,2,N), 1, 5)
age = np.clip(rng.gamma(2.2, 4.8, N), 0, 40).round(0).astype(int)
loc = 3.5+0.5*(np.array([zones[z][2] for z in zn])-3.5) + rng.normal(0,.45,N)
loc = np.clip(loc.round(1),1,5)
dist = np.array([zones[z][3] for z in zn]) + rng.normal(0,2.2,N)
dist = np.clip(dist,0.5,35).round(1)
prem = np.array([zones[z][0] for z in zn])
price = (1.4 + prem + 3.85*area/100 + 5.4*bed - 0.65*age + 4.1*loc + rng.normal(0,6.0,N))
df = pd.DataFrame({'House_ID':np.arange(1,N+1),'Price_Lakhs':price.round(2),'Area_SqFt':area.astype(int),
  'Bedrooms':bed,'Bathrooms':bath,'Property_Age':age,'Distance_to_Centre':dist,
  'Locality_Score':loc,'Location_Zone':zn,'Zone_Type':[zones[z][1] for z in zn]})
df.to_csv('housing.csv', index=False)
print(df.describe().T[['mean','50%','std']])
