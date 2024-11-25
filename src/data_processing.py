import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import config as cf

def load_data():
    combined_df = pd.read_csv(cf.COMBINED_FILEPATH)
    df_rate = pd.read_csv(cf.FILEPATH)
    return combined_df, df_rate

def preprocess_data(combined_df, target):
    train_df = combined_df.drop(columns=["month", 'year', 'MeasurementCount'], axis=1)
    train_df = train_df.select_dtypes(['int64', 'float64'])
    return train_df

def plot_correlation_matrix(df, target, output_path):
    corr_matrix = pd.concat([df, target], axis=1).select_dtypes(["int"]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.savefig(output_path)
    plt.show()