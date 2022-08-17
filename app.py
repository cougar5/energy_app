import joblib
from flask import Flask, request, render_template, Markup
import numpy as np

app = Flask("__name__")

def checkglaze(glazed, glazearea, glazedir):
    if glazed == "False":
        glaze = [0, 0, 0, 0, 0, 0]
    else:
        glaze = [glazearea, *glazedir]
    return glaze

def makepredict(entry):
    coolpre = joblib.load("Cmodel")
    heatpre = joblib.load("Hmodel")
    
    Cresult = round(float(coolpre.predict(entry)),2)
    Hresult = round(float(heatpre.predict(entry)),2)
    return Cresult,Hresult

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def getdata():
    inGlaze = {"Eastward":0, "Northward":0, "Southward":0, "Uniform":0, "Westward":0}
    inDirection = {"East":0, "North":0, "South":0, "West":0}

    #Read area and orientation values
    Relative_Compactness = float(request.values.get('Relative_Compactness'))/100
    Surface_Area = float(request.values.get('Surface_Area'))
    Wall_Area = float(request.values.get('Wall_Area'))
    Roof_Area = float(request.values.get('Roof_Area'))
    Overall_Height = float(request.values.get('Overall_Height'))
    inDirection[request.form.get('Orientation')] = 1
    dir = [*inDirection.values()]

    #Check glazing
    Glazed = request.values.get('Glazed')
    Glazing_Area = float(request.values.get('Glazing_Area'))
    inGlaze[request.form.get('Glazing_dir')] = 1
    glaze = checkglaze(Glazed, Glazing_Area, [*inGlaze.values()])

    entrydata = [Relative_Compactness,Surface_Area, Wall_Area, Roof_Area, Overall_Height] + glaze + dir
    predictdata = np.array(entrydata, dtype=object).reshape((-1,15))
    cool,heat = makepredict(predictdata)

    record = entrydata + [cool, heat] #For further function in same format of training dataset
    result = Markup(f'Cooling Load: {cool}kWh<br>Heating Load: {heat}kWh')
    return render_template("result.html", result=result)

app.run()