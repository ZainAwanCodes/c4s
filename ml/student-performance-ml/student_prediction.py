# Project: Student Pass/Fail Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=== STEP 1: Understand the Dataset ===")
# We load the dataset using Pandas read_csv. The separator is ';' instead of a comma.
df = pd.read_csv("student-mat.csv", sep=";")

# Inspecting the dataset:
print("\n--- df.head() ---")
print(df.head()) # Shows the first 5 rows of the dataset.

print("\n--- df.tail() ---")
print(df.tail()) # Shows the last 5 rows of the dataset.

print("\n--- df.shape ---")
print(df.shape)  # Shows the number of rows and columns (rows, columns).

print("\n--- df.columns ---")
print(df.columns) # Lists all the column names in the dataset.

print("\n--- df.info() ---")
df.info() # Shows data types, non-null counts, and memory usage.

print("\n--- df.describe() ---")
print(df.describe()) # Provides summary statistics (mean, min, max, etc.) for numerical columns.

print("\n--- df.isnull().sum() ---")
print(df.isnull().sum()) # Checks for missing values in each column.

print("\n--- df.duplicated().sum() ---")
print(df.duplicated().sum()) # Checks for identical/duplicate rows.


print("\n=== STEP 2: Clean the Data Using Pandas ===")
# Our dataset doesn't have missing values or duplicates right now based on our checks.
# If it did, we could use df.dropna() to remove missing values or df.drop_duplicates().
# We will create our target column "Result" based on the "G3" column.
print("Creating the 'Result' target column based on 'G3' (Pass >= 10, Fail < 10)")
df["Result"] = df["G3"].apply(lambda x: "Pass" if x >= 10 else "Fail")
print(df[["G3", "Result"]].head())


print("\n=== STEP 3: Select Features ===")
# We select a subset of meaningful columns to use as input features (X) to predict the output (y).
features = ["studytime", "failures", "absences", "Medu", "Fedu", "goout", "health", "Dalc", "Walc"]

# X represents the input features (the data the model uses to learn).
X = df[features]
# y represents the target (what we want the model to predict).
y = df["Result"]

print("Features (X):", X.columns.tolist())
print("Target (y):", y.name)


print("\n=== STEP 4: Convert Categorical Data ===")
# Our selected features are already numeric (e.g., studytime is 1-4). 
# If we had text categories like "yes" or "no", we would need to convert them to numbers (like 1 and 0)
# because Scikit-learn algorithms only understand numbers.
# We can do this with pd.get_dummies(), which creates new columns for each category with 1s and 0s.
print("No categorical conversion needed for our selected features, as they are already numeric representations.")


print("\n=== STEP 5: Train/Test Split ===")
# We split our data into two parts: training data (to teach the model) and testing data (to test the model).
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# X_train: The features used to train the model.
# y_train: The actual answers (targets) for the training data.
# X_test: The features used to test the model's accuracy on unseen data.
# y_test: The actual answers for the testing data to compare against the model's predictions.
# test_size=0.2: We use 20% of the data for testing, and 80% for training.
# random_state=42: Ensures we get the exact same split every time we run the code.
# stratify=y: Ensures the ratio of Pass/Fail is the same in both the training and testing sets.
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")


print("\n=== STEP 6: Train the Model ===")
# We create a Logistic Regression model (a simple algorithm for classification tasks).
model = LogisticRegression(max_iter=1000)

# model.fit() is where the actual learning happens. 
# The model looks at the training features (X_train) and the correct answers (y_train) 
# and finds the mathematical relationship between them.
model.fit(X_train, y_train)
print("Model trained successfully!")


print("\n=== STEP 7: Make Predictions ===")
# Now we ask the model to predict the results for our testing data.
y_pred = model.predict(X_test)

# Let's print some of the actual values vs what our model predicted
print("Sample Predictions:")
results_comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
for index, row in results_comparison.head().iterrows():
    print(f"Actual: {row['Actual']} | Predicted: {row['Predicted']}")


print("\n=== STEP 8: Evaluate the Model ===")
# We evaluate how well the model did by comparing its predictions (y_pred) to the real answers (y_test).

# Accuracy: The percentage of correct predictions overall.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nConfusion Matrix:")
# Confusion Matrix: A table showing True Positives, True Negatives, False Positives, and False Negatives.
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
# Classification Report: Shows Precision, Recall, and F1-score for each class.
# - Precision: When the model predicts "Pass", how often is it actually "Pass"?
# - Recall: Out of all the actual "Pass" students, how many did the model find?
# - F1-score: A balance between Precision and Recall.
print(classification_report(y_test, y_pred))


print("\n=== STEP 9: Predict a New Student ===")
print("Please enter the following student details to get a prediction:")

try:
    studytime = int(input("Study time (1-4): "))
    failures = int(input("Number of past class failures (0-3): "))
    absences = int(input("Number of school absences (0-93): "))
    Medu = int(input("Mother's education level (0-4): "))
    Fedu = int(input("Father's education level (0-4): "))
    goout = int(input("Going out with friends (1-5): "))
    health = int(input("Current health status (1-5): "))
    Dalc = int(input("Workday alcohol consumption (1-5): "))
    Walc = int(input("Weekend alcohol consumption (1-5): "))

    new_student = pd.DataFrame({
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "goout": [goout],
        "health": [health],
        "Dalc": [Dalc],
        "Walc": [Walc]
    })
    
    # We ask our trained model to predict the outcome for this new student.
    prediction = model.predict(new_student)
    print("\n--- New Student Data ---")
    print(new_student.to_string(index=False))
    print(f"\n=> PREDICTED RESULT: {prediction[0]} <=")
except ValueError:
    print("Invalid input! Please enter numbers only.")
