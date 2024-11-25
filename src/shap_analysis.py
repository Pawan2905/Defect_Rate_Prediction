import shap
import matplotlib.pyplot as plt

def analyze_shap(model, X_test, output_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(output_path)