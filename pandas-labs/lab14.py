import pandas as pd
data = { 
"Name":["Ali","Sara","Usman","Hina","Bilal"], 
"Hours":[2,5,None,6,3], 
"Attendance":[60,None,50,92,70], 
"Marks":[48,78,40,None,62] 
} 
missing = pd.DataFrame(data)
print(missing.isna().sum())
print(missing['Hours'].mean())
missing['Hours'] = missing['Hours'].fillna(missing['Hours'].mean())
print(missing)
missing_dropped = missing.copy().dropna()
print(f"Original rows: {len(missing)}, After dropna: {len(missing_dropped)}")
