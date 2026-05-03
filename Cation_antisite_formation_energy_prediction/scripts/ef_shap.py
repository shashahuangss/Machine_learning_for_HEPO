import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
import shap


# Plot style
mpl.rcParams['font.family'] = 'Calibri'
mpl.rcParams['font.size'] = 32
mpl.rcParams['axes.titlesize'] = 32
mpl.rcParams['axes.labelsize'] = 32
mpl.rcParams['xtick.labelsize'] = 32
mpl.rcParams['ytick.labelsize'] = 32
mpl.rcParams['legend.fontsize'] = 32


def load_data(data_path="../data/Ef_train_data.xlsx", scaler_path="../models/scaler.joblib"):
    """Load raw data and scaler, return scaled features and feature names."""
    data = pd.read_excel(data_path, sheet_name="Sheet1")
    data = data.dropna(axis=1, how="all").drop_duplicates().dropna()

    X = data.iloc[:, :62]
    feature_names = X.columns.tolist()

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    return X_scaled, feature_names


def run_shap_analysis(model_path="../models/rf_model.joblib",
                      data_path="../data/Ef_train_data.xlsx",
                      scaler_path="../models/scaler.joblib",
                      output_excel="../results/SHAP_values.xlsx",
                      output_plot="../results/SHAP_summary_plot.png",
                      max_display=10):
    """Compute SHAP values for the trained RF model and generate summary plot."""
    model = joblib.load(model_path)
    X_scaled, feature_names = load_data(data_path, scaler_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_excel(output_excel, index=False)
    print(f"SHAP values saved to {output_excel}")

    # Summary plot
    shap.summary_plot(
        shap_values,
        X_scaled,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(12, 8)

    # Override fonts to Arial 26pt
    for ax in fig.axes:
        ax.title.set_fontsize(26)
        ax.title.set_fontfamily('Arial')
        ax.xaxis.label.set_fontsize(26)
        ax.xaxis.label.set_fontfamily('Arial')
        ax.yaxis.label.set_fontsize(26)
        ax.yaxis.label.set_fontfamily('Arial')
        for label in ax.get_xticklabels():
            label.set_fontsize(26)
            label.set_fontfamily('Arial')
        for label in ax.get_yticklabels():
            label.set_fontsize(26)
            label.set_fontfamily('Arial')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Summary plot saved to {output_plot}")
    plt.show()


def main():
    run_shap_analysis()


if __name__ == "__main__":
    main()
