import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn import preprocessing


dataTrain = pd.read_csv("50_Startups.csv")

le = preprocessing.LabelEncoder()
le.fit(dataTrain['State'])
print(list(le.classes_))
print(le.transform(dataTrain['State']))
#dataTest = pd.read_csv("dataTest.csv")
# print df.head()

#x_train = dataTrain[['R&D Spend', 'Administration']]
#y_train = dataTrain['CompressibilityFactor(Z)']



x_train=dataTrain.iloc[2:38,0:3].values
y_train=dataTrain.iloc[2:38,4].values

x_test=dataTrain.iloc[39:51,0:3].values
y_test=dataTrain.iloc[39:51,4].values

#x_test = dataTest[['Temperature(K)', 'Pressure(ATM)']]
#y_test = dataTest['CompressibilityFactor(Z)']

ols = linear_model.LinearRegression()
#ols=SVR(kernel='linear',degree=1)
ols.fit(x_train, y_train)
pred=ols.predict(x_test)


#print(r2_score(y_test,pred))
print(r2_score(y_test,pred))
print (pred)