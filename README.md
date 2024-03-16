# Insurance-Premium-Prediction

## Overview
This project focuses on predicting healthcare insurance premiums for individuals based on various factors such as age, BMI, smoking status, and other demographic variables. By leveraging machine learning techniques, the goal is to develop accurate predictive models that can assist insurance providers in estimating insurance premiums more effectively.

## Dataset Source
The dataset used in this project was obtained from Kaggle and can be found [here](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction).

## Project Structure
- **Data Cleaning and Preprocessing**: Conducted thorough cleaning and preprocessing of the dataset to ensure data quality and integrity.
- **Exploratory Data Analysis (EDA)**: Utilized Python libraries including NumPy, Pandas, Matplotlib, and Seaborn to perform EDA, visualizing relationships between variables and identifying patterns.
- **Feature Engineering**: Implemented feature engineering techniques to extract meaningful insights from the data and prepare it for modeling.
- **Model Building**: Developed predictive models using Polynomial Regression, Decision Tree Regression, and Random Forest Regression to forecast medical expenses.
- **Evaluation Metrics**: Assessed model performance using R-squared scores and Root Mean Squared Error (RMSE) to gauge accuracy and precision.

## Tools and Libraries
- **Python**: Utilized Python programming language for data analysis, preprocessing, and modeling.
- **NumPy and Pandas**: Used for data manipulation and analysis.
- **Matplotlib and Seaborn**: Employed for data visualization and generating insights.
- **Scikit-learn**: Implemented machine learning algorithms for model building and evaluation.

## Results
- Age has a positive correlation with expenses, indicating that older individuals tend to have higher medical expenses.
- Smokers demonstrate significantly higher expenses compared to non-smokers.
- For smokers, expenses increase with BMI, while non-smokers exhibit a concentration of data points within a specific BMI and expense range.
- The correlation heatmap reveals predominantly weak relationships among variables, with the highest correlation observed between age and expenses.
- Despite similar R-squared scores, the Random Forest Regression model performs slightly better in predicting insurance expenses compared to Polynomial and Decision Tree Regression models, as indicated by lower RMSE values.
