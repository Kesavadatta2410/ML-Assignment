# ML-Assignment
In collaboration with my team members, I completed four tasks that involved implementing machine learning models in real-world applications. These projects provided valuable insights and practical experience applying machine learning techniques to solve complex problems.


Overview
This repository contains Python scripts for various machine-learning tasks. Each script demonstrates different aspects of data preprocessing, model training, and evaluation. Below is a brief description of each script and the expected outputs.

Scripts


1. task2.py
    This script performs the following tasks:
    Loads a dataset from a CSV file.
    Preprocesses the data by handling missing values and encoding categorical variables.
    Trains a Random Forest Regressor model to predict vehicle commissions.
    Evaluate the model using Mean Absolute Error (MAE).
    Visualizes the distribution of actual and predicted commissions.
    Generates a scatter plot to compare actual vs. predicted commissions.
    
    Expected Outputs:
    Mean Absolute Error (MAE): A numerical value indicating the model's performance.
    Mean Absolute Error: 0.1708046875000062
    Predicted Commissions: A table showing the predicted commissions for the first few vehicles.
    Predicted CO2 Emissions (Commissions) by Make and Model: A table summarizing the predicted commissions by vehicle make and model.

    Graphs:
    Distribution of Actual and Predicted Commissions.
    Scatter plot of Actual vs. Predicted Commissions.
    Output Graphs:
    Distribution of Actual and Predicted Commissions
    Scatter plot of Actual vs. Predicted Commissions

3. task3.py
This script demonstrates the use of a Naive Bayes classifier for a categorical dataset:

Encodes categorical data using LabelEncoder.

Trains a Categorical Naive Bayes model.

Predicts the class probabilities and the class for a new data point.

Expected Outputs:
Class Probabilities: Probabilities for each class (e.g., '+' or '-').

Predicted Class: The predicted class for the new data point.

Example Output:
Copy
Class Probabilities:
Class 0 (-): 0.1234
Class 1 (+): 0.8766

Predicted Class: (+)
3. task4.1.py
This script demonstrates polynomial regression:

Generates synthetic data for tractor age and maintenance cost.

Fits a polynomial regression model to the data.

Visualizes the actual data points and the predicted regression curve.

Expected Outputs:
Graph: A scatter plot of tractor age vs. maintenance cost with the predicted regression curve.

Output Graph:
Tractor Age vs Maintenance Cost

4. task4.2.py
This script extends task4.1.py by evaluating the polynomial regression model:

Splits the data into training and testing sets.

Fits a polynomial regression model.

Evaluates the model using Mean Squared Error (MSE) and R-squared (R²).

Expected Outputs:
Mean Squared Error (MSE): A numerical value indicating the model's performance.

R-squared (R²): A numerical value indicating the goodness of fit.

Example Output:
Copy
Mean Squared Error: 1234.56
R-squared: 0.9876
Libraries Required
Ensure you have Python installed along with the required libraries:
pandas
NumPy
matplotlib
seaborn
scikit-learn
