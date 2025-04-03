# Regression Machine Learning Models

This repository contains various regression machine learning models implemented in Python. It serves as a collection of different approaches to solving regression problems using various algorithms and techniques.

## Follow

### You can follow the project in order of commit

## Features
- Implementation of multiple regression algorithms.
- Performance evaluation using various metrics.
- Data preprocessing and feature scaling.
- Hyperparameter tuning using GridSearchCV.
- Visualization of results using Matplotlib.

## Algorithms Used
- Linear Regression
- Decision Tree Regressor (CART)
- K-Nearest Neighbors (KNN) Regressor
- Multi-Layer Perceptron (MLP) Regressor
- Random Forest Regressor
- Gradient Boosting Regressor (GBM)
- LightGBM Regressor
- XGBoost Regressor
- Support Vector Regressor (SVR)

## Dependencies
The following Python libraries are used in this repository:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
```

## Installation
To use the models, install the required dependencies using pip:
```bash
pip install numpy matplotlib pandas scikit-learn lightgbm xgboost
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/CivitCihan/ML_Regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML_Regression
   ```
3. Run the desired script to train and evaluate models.

## Contribution
Feel free to contribute by submitting issues or pull requests to improve existing models or add new regression techniques.
