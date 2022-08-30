from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def home(request):
    return render(request, 'home.html')
def Diabetes(request):
    return render(request, 'Diabetes.html')

def result(request):
    dataset = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
    zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
    for column in zero_not_accepted:
        dataset[column] = dataset[column].replace(0,np.NaN)
        mean = int(dataset[column].mean(skipna=True))
        dataset[column] = dataset[column].replace(np.NaN,mean)
    X = dataset.iloc[:,0:8]
    y = dataset.iloc[:,8]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.2)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
    classifier.fit(X_train, y_train)
    val1 = float(request.GET['Pregnancies'])
    val2 = float(request.GET['Glucose'])
    val3 = float(request.GET['BloodPressure'])
    val4 = float(request.GET['SkinThickness'])
    val5 = float(request.GET['Insulin'])
    val6 = float(request.GET['BMI'])
    val7 = float(request.GET['DiabetesPedigreeFunction'])
    val8 = float(request.GET['Age'])
    pred = sc_X.transform([[val1,val2, val3, val4, val5,val6, val7, val8]])
    x = classifier.predict(pred)
    result1 = ""
    if x == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"



    return render(request, 'Diabetes.html',{"result2":result1})
