# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Load dataset
data = pd.read_csv("50_Startups.csv")

X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

# ElasticNet Regression
en_model = ElasticNet()
en_params = {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
en_grid = GridSearchCV(en_model, en_params, cv=5)
en_grid.fit(X_train, y_train)
en_pred = en_grid.predict(X_test)
en_mae = mean_absolute_error(y_test, en_pred)
en_mse = mean_squared_error(y_test, en_pred)
en_r2 = r2_score(y_test, en_pred)

# KNN Regression
knn_model = KNeighborsRegressor()
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(knn_model, knn_params, cv=5)
knn_grid.fit(X_train_scaled, y_train)
knn_pred = knn_grid.predict(X_test_scaled)
knn_mae = mean_absolute_error(y_test, knn_pred)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_r2 = r2_score(y_test, knn_pred)

# Save figures for later use
plt.figure(figsize=(18, 5))

# Linear Regression plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, lr_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Linear Regression')

# ElasticNet Regression plot
plt.subplot(1, 3, 2)
plt.scatter(y_test, en_pred, alpha=0.7, color='r')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('ElasticNet Regression')

# KNN Regression plot
plt.subplot(1, 3, 3)
plt.scatter(y_test, knn_pred, alpha=0.7, color='g')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('KNN Regression')

plt.tight_layout()
plt.savefig('regression_plots.png')  # Save the figure for later use

# Combined residuals plot
plt.figure(figsize=(12, 8))

plt.scatter(y_test, lr_pred, alpha=0.6, color='b', s=50, label='Linear Regression')
plt.scatter(y_test, en_pred, alpha=0.6, color='r', s=50, label='ElasticNet Regression')
plt.scatter(y_test, knn_pred, alpha=0.6, color='g', s=50, label='KNN Regression')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel('Actual Profit', fontsize=14)
plt.ylabel('Predicted Profit', fontsize=14)
plt.title('Actual vs Predicted Profit', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('combined_residuals.png')  # Save the figure for later use

# Streamlit app
def main():
    st.title('Profit Prediction and Model Evaluation')

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Prediction", "Model Evaluation"])

    if page == "Prediction":
        st.header('Profit Prediction')
        st.write("Enter the values to predict profit:")

        rd_spend = st.number_input('R&D Spend')
        admin = st.number_input('Administration')
        marketing = st.number_input('Marketing Spend')

        prediction_options = st.selectbox("Select a regression model", ["Linear Regression", "ElasticNet Regression", "KNN Regression"])

        if prediction_options == "Linear Regression":
            prediction = lr_model.predict([[rd_spend, admin, marketing]])
        elif prediction_options == "ElasticNet Regression":
            prediction = en_grid.predict([[rd_spend, admin, marketing]])
        else:  # KNN Regression
            scaled_input = scaler.transform([[rd_spend, admin, marketing]])
            prediction = knn_grid.predict(scaled_input)

        st.subheader("Profit Prediction:")
        st.write(f"The predicted profit is: ${prediction[0]:.2f}")

    elif page == "Model Evaluation":
        st.header('Model Performance Metrics')
        st.write("Here are the performance metrics for each regression model:")

        st.write("**Linear Regression**")
        st.write(f"MAE: {lr_mae}, MSE: {lr_mse}, R2: {lr_r2}")

        st.write("**ElasticNet Regression**")
        st.write(f"MAE: {en_mae}, MSE: {en_mse}, R2: {en_r2}")

        st.write("**KNN Regression**")
        st.write(f"MAE: {knn_mae}, MSE: {knn_mse}, R2: {knn_r2}")

        st.header('Regression Model Plots')
        st.write("**Individual Regression Plots**")

        st.image('regression_plots.png', use_column_width=True)

        st.write("**Combined Residuals Plot**")
        st.image('combined_residuals.png', use_column_width=True)

if __name__ == '__main__':
    main()
