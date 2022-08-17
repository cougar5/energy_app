import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# Data for prediction
energy = pd.read_csv("energy_efficiency_data.csv")
print(energy.head())

# For Orientation process
Direction = {2:"North", 3:"East", 4:"South", 5:"West"}

# For Glazing_Area_Distribution process
Glaze = {0:'NotGlazed', 1:"Uniform", 2:"Northward", 3:"Eastward", 4:"Southward", 5:"Westward"}

data = energy.drop(["Cooling_Load", "Heating_Load"], axis=1)
data = data.replace({"Orientation": Direction})
data = data.replace({"Glazing_Area_Distribution": Glaze})
print(data.head())

energy_direction = pd.get_dummies(data['Orientation'])
print(energy_direction.head())
energy_glazing = pd.get_dummies(data['Glazing_Area_Distribution']).drop('NotGlazed', axis = 1)
print(energy_glazing.head())

X = pd.concat([data,energy_glazing,energy_direction],axis = 1)
X.drop(columns=["Orientation","Glazing_Area_Distribution"], inplace = True)
print(X.head())

yc = energy[["Cooling_Load"]]
print(yc.head())
yh = energy[["Heating_Load"]]
print(yh.head())

X = np.array(X)
yc = np.array(yc)
yh = np.array(yh)

#Build models
Cmodel = xgb.XGBRegressor()
Hmodel = xgb.XGBRegressor()

Cmodel.fit(X,yc.reshape(-1,))
joblib.dump(Cmodel,"Cmodel")

Hmodel.fit(X,yh.reshape(-1,))
joblib.dump(Hmodel,"Hmodel")