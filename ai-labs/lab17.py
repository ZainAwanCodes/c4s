import pandas as pd 
data = { 
"StudyHours":[2,5,1,6,3,7,4,8], 
"Attendance":[60,85,50,92,70,95,75,98], 
"Assignments":[45,75,35,88,60,92,68,95], 
"FinalMarks":[48,78,40,90,62,96,70,98] 
} 
df = pd.DataFrame(data)

X = df[['StudyHours', 'Attendance', 'Assignments']]
y = df['FinalMarks']
print(X.shape)
print(y.shape)
