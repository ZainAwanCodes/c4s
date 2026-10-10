import pandas as pd
from sklearn.linear_model import LinearRegression

data = { 
"StudyHours":[2,5,1,6,3,7,4,8], 
"Attendance":[60,85,50,92,70,95,75,98], 
"Assignments":[45,75,35,88,60,92,68,95], 
"FinalMarks":[48,78,40,90,62,96,70,98] 
} 
df = pd.DataFrame(data)

X = df[["StudyHours","Attendance","Assignments"]]
y = df["FinalMarks"] 
model = LinearRegression() 
model.fit(X, y) 
new_data = pd.DataFrame({ 
"StudyHours":[8], 
"Attendance":[99], 
"Assignments":[87] 
}) 
prediction = model.predict(new_data) 
print("Prediction:", prediction)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)