import pandas as pd 
data = { 
"Name":["Ali","Sara","Usman","Hina","Bilal","Noor"], 
"Hours":[2,5,1,6,3,7], 
"Attendance":[60,85,50,92,70,95], 
"Marks":[48,78,40,90,62,96] 
} 
df = pd.DataFrame(data)
print(df[['Name','Marks']])
marks=df['Marks']
print(marks.mean())

