import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "StudyHours": [2, 5, 1, 6, 3, 7, 4, 8],
    "Attendance": [60, 85, 50, 92, 70, 95, 75, 98],
    "Assignments": [45, 75, 35, 88, 60, 92, 68, 95],
    "FinalMarks": [48, 78, 40, 90, 62, 96, 70, 98]
}

df = pd.DataFrame(data)

# Features
X = df[["StudyHours", "Attendance", "Assignments"]]

# Target
y = df["FinalMarks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Take user input
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
assignments = float(input("Enter assignment marks: "))

# Create new student
student = pd.DataFrame({
    "StudyHours": [hours],
    "Attendance": [attendance],
    "Assignments": [assignments]
})

# Predict
prediction = model.predict(student)[0]

print(f"\nPredicted Final Marks: {prediction:.2f}")