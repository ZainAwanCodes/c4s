import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)

import kagglehub
import os

# 1. Load data
path = kagglehub.dataset_download("jacksondivakarr/student-classification-dataset")
csv_path = os.path.join(path, "student.csv")
df = pd.read_csv(csv_path)

# 2. Inspect
print(df.head())
print(df.shape)
print(df.isnull().sum())

# 3. Clean and encode features
# Weekly_Study_Hours is numeric
df["Weekly_Study_Hours"] = df["Weekly_Study_Hours"].fillna(df["Weekly_Study_Hours"].median())

# Attendance is categorical ("Always", "Never", "Sometimes", etc.), we can encode it
df["Attendance"] = df["Attendance"].astype('category').cat.codes

# Let's also include more features to improve accuracy
categorical_cols = ["Reading", "Notes", "Listening_in_Class", "Project_work"]
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# 4. Select features and target
X = df[["Weekly_Study_Hours", "Attendance"] + categorical_cols]
y = df["Grade"]

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=42,stratify=y)

# 6. Create and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Predict
predictions = model.predict(X_test)

# 8. Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification report:")
print(classification_report(y_test, predictions))
