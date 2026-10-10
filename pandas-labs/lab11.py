import pandas as pd
data = { 
"Name":["Ali","Sara","Usman","Hina","Bilal","Noor"], 
"Hours":[2,5,1,6,3,7], 
"Attendance":[60,85,50,92,70,95], 
"Marks":[48,78,40,90,62,96] 
} 
df = pd.DataFrame(data)
print(df[(df['Marks']<=50) | (df['Attendance']<60)])