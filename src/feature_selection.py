from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def select_features(train_df, target, n_features=50):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(train_df, target)
    selected_features = train_df.columns[rfe.support_]
    return train_df[selected_features]