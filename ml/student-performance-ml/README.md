# Student Pass/Fail Prediction

This is a beginner-level Machine Learning project designed to predict whether a student will Pass or Fail based on their academic and personal features. 

The project strictly uses **Pandas** for data manipulation and **Scikit-learn** for Machine Learning.

## Project Structure

- `student-mat.csv`: The dataset containing student performance data.
- `student_prediction.py`: The Python script containing the full ML pipeline.
- `README.md`: This documentation file.

## The Machine Learning Workflow

The Python script follows a simple logical pipeline:

1. **Understand the Dataset**: We use Pandas functions like `df.head()`, `df.info()`, and `df.describe()` to get a feel for our data's structure and statistics. We also check for missing values.
2. **Clean the Data**: We prepare our target variable `Result` based on the final grade `G3` (10 or above is a Pass). 
3. **Select Features**: We separate our data into `X` (the input features like study time, absences, health) and `y` (the target we want to predict, which is the Pass/Fail `Result`).
4. **Convert Categorical Data**: ML models only understand numbers. If we had text features, we'd use Pandas to convert them to numeric representations.
5. **Train/Test Split**: We split the data into a training set (80%) to teach the model, and a testing set (20%) to evaluate how well it learned.
6. **Train the Model**: We use a `LogisticRegression` algorithm. By calling `model.fit()`, the algorithm learns the patterns between the student features and their Pass/Fail outcome.
7. **Make Predictions**: We use the testing data to generate predictions (`y_pred`) to see what the model thinks the outcomes should be.
8. **Evaluate the Model**: We compare the model's predictions to the actual answers using metrics like Accuracy (percentage correct), a Confusion Matrix, and a Classification Report (showing Precision and Recall).
9. **Predict a New Student**: Finally, we create hypothetical data for a new student and use our trained model to predict whether they will Pass or Fail!

## How to Run
Ensure you have Pandas and Scikit-learn installed:
```bash
pip install pandas scikit-learn
```

Run the script:
```bash
python student_prediction.py
```
