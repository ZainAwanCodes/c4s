import pandas as pd
from sklearn.linear_model import LinearRegression

# 1- Read CSV
df = pd.read_csv("week2-day2\Olympic_Athlete_Bio.csv")

# 2- Clean it using dropna
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df_clean = df.dropna(subset=['height', 'weight'])

# 3- Use Height as input x and weight as output Y
X = df_clean[['height']]
Y = df_clean['weight']

# 4- Use linear regression of sklearn to build model
model = LinearRegression()
model.fit(X, Y) 

# 5- Take height from user and display predicted weight
user_height = float(input("Enter height to predict weight: "))
predicted_weight = model.predict([[user_height]])
print(f"Predicted weight: {predicted_weight[0]:.2f}")
