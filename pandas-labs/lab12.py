import pandas as pd
data = { 
"Name":["Ali","Sara","Usman","Hina","Bilal","Noor"], 
"Hours":[2,5,1,6,3,7], 
"Attendance":[60,85,50,92,70,95], 
"Marks":[48,78,40,90,62,96] 
} 
df = pd.DataFrame(data)
print(df.iloc[0])
print(df.iloc[0:3])
print(df.loc[df['Marks'].idxmax()])
print(df.sort_values(by='Marks'))
print(df.sort_values(by='Marks', ascending=False))
print(df.sort_values(by='Attendance', ascending=False))
