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


def load_data(data_path, scaler_path='../models/standard_scaler.joblib'):
    """Load raw data and the fitted scaler, return scaled features and feature names."""
    df = pd.read_excel(data_path)
    X = df.iloc[:, :36].values
    feature_names = df.columns[:36].tolist()

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    return X_scaled, feature_names


def run_shap_analysis(model_path='../models/random_forest_model.joblib',
                      data_path='../data/phase_train_data.xlsx',
                      scaler_path='../models/standard_scaler.joblib',
                      output_excel='../results/SHAP_values.xlsx',
                      output_plot='../results/SHAP_summary_plot.png',
                      max_display=10):
    """Compute SHAP values for the positive class and generate a summary plot."""
    model = joblib.load(model_path)
    X_scaled, feature_names = load_data(data_path, scaler_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)

    # Extract SHAP values for the positive class (binary classification)
    shap_values_pos = shap_values.values[..., 1]
    print(f"SHAP values shape: {shap_values_pos.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Save SHAP values to Excel
    if shap_values_pos.shape[1] == len(feature_names):
        shap_df = pd.DataFrame(shap_values_pos, columns=feature_names)
        shap_df.to_excel(output_excel, index=False)
        print(f"SHAP values saved to {output_excel}")
    else:
        print(f"Warning: shape mismatch ({shap_values_pos.shape} vs {len(feature_names)} features), skipping Excel export.")

    # Summary plot
    shap.summary_plot(
        shap_values_pos,
        X_scaled,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(12, 8)

    # Override fonts to Arial 26pt for publication quality
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


if __name__ == '__main__':
    main()
