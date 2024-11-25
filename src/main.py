import data_processing as dp # load_data, preprocess_data, plot_correlation_matrix
import feature_selection as fs  #select_features
import model_training as mt # split_data, train_linear_regression, train_random_forest, train_xgboost, evaluate_model
import hyperparameter_tuning as ht #  random_search_rf, random_search_xgb
import shap_analysis as sh # analyze_shap
import pandas as pd
import numpy as np
import pickle
import sys
import os

# Load data
combined_df, df_rate = dp.load_data()
target = df_rate[['Defect_Rate']]

# Plot correlation matrix
dp.plot_correlation_matrix(combined_df, target, "../output/correlationplot.png")

# Preprocess data
train_df = dp.preprocess_data(combined_df, target)

# Feature selection
X = fs.select_features(train_df, target)
y = target

# Split data
X_train, X_test, y_train, y_test = mt.split_data(X, y)

# Train models
lin_reg = mt.train_linear_regression(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
rf_model = mt.train_random_forest(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
xg_model = mt.train_xgboost(X_train, y_train)
y_pred_xg = xg_model.predict(X_test)

# Evaluate models
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'Metrics': [
        mt.evaluate_model(y_test, y_pred_lin),
        mt.evaluate_model(y_test, y_pred_rf),
        mt.evaluate_model(y_test, y_pred_xg),
    ]
})
print(metrics_df)
metrics_df.to_csv("../output/baseline_models.csv",index=False)

# Hyperparameter tuning
rf_param_dist ={
    'n_estimators': np.arange(50, 301, 50),  # Number of trees
    'max_depth': [None, 10, 20, 30, 40, 50],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at a leaf node
    'max_features': [None, 'sqrt', 'log2'],  # Use None instead of 'auto'
    'bootstrap': [True, False],  # Whether bootstrap samples are used
}
best_rf_model = ht.random_search_rf(X_train, y_train, rf_param_dist)
xg_param_dist = {'n_estimators': np.arange(50, 301, 50),  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Learning rate (shrinkage)
    'max_depth': [3, 6, 10, 12, 15],  # Maximum depth of the tree
    'min_child_weight': [1, 3, 5, 7],  # Minimum sum of instance weight (hessian) in a child
    'subsample': [0.5, 0.7, 1.0],  # Proportion of training data to use in each tree
    'colsample_bytree': [0.5, 0.7, 1.0],  # Proportion of features to use in each tree
    'gamma': [0, 0.1, 0.3, 0.5],  # Minimum loss reduction required to make a further partition
    'scale_pos_weight': [1, 2, 5],  # Balancing of positive and negative weights
}
best_xg_model = ht.random_search_xgb(X_train, y_train, xg_param_dist)

# Evaluate tuned models
y_pred_rf_tuned = best_rf_model.predict(X_test)
y_pred_xg_tuned = best_xg_model.predict(X_test)

rf_metrics_tuned = mt.evaluate_model(y_test, y_pred_rf_tuned)
xg_metrics_tuned = mt.evaluate_model(y_test, y_pred_xg_tuned)

# Save tuned metrics to Excel
tuned_metrics_df = pd.DataFrame({
    'Model': ['Random Forest (Tuned)', 'XGBoost (Tuned)'],
    'R2': [rf_metrics_tuned[0], xg_metrics_tuned[0]],
    'MSE': [rf_metrics_tuned[1], xg_metrics_tuned[1]],
    'RMSE': [rf_metrics_tuned[2], xg_metrics_tuned[2]],
    'MAPE': [rf_metrics_tuned[3], xg_metrics_tuned[3]]
})
tuned_metrics_df.to_excel("../output/tuned_RF_LSTM_metrics.xlsx", index=False)


# Save models
with open("../saved_model/best_rf_model.pkl", 'wb') as f:
    pickle.dump(best_rf_model, f)
with open("../saved_model/best_xg_model.pkl", 'wb') as f:
    pickle.dump(best_xg_model, f)

# SHAP Analysis
sh.analyze_shap(best_rf_model, X_test, "../output/shap_feature_importance.png")
