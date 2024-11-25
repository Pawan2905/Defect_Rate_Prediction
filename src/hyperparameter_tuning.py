from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def random_search_rf(X_train, y_train, param_dist, n_iter=100):
    rf_random_search = RandomizedSearchCV(
        RandomForestRegressor(),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rf_random_search.fit(X_train, y_train)
    return rf_random_search.best_estimator_

def random_search_xgb(X_train, y_train, param_dist, n_iter=100):
    xg_random_search = RandomizedSearchCV(
        xgb.XGBRegressor(),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    xg_random_search.fit(X_train, y_train)
    return xg_random_search.best_estimator_
