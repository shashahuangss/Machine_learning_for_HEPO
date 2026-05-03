import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path="../data/Ef_train_data.xlsx"):
    """Load Excel data, drop empty columns, duplicates, and NaN rows."""
    data = pd.read_excel(path, sheet_name="Sheet1")
    data = data.dropna(axis=1, how="all").drop_duplicates().dropna()
    X = data.iloc[:, :62]
    y = data["Formation energy"]
    return X, y


def compute_pearson_feature_target(X, y, output_excel="../results/pearson_feature_vs_formation_energy.xlsx"):
    """Compute Pearson correlation between each feature and target."""
    results = []
    for feature in X.columns:
        r, p = pearsonr(X[feature], y)
        results.append({
            "Feature": feature,
            "Pearson_r": r,
            "p_value": p,
            "|r|": abs(r),
        })

    df = pd.DataFrame(results).sort_values("|r|", ascending=False)
    df.to_excel(output_excel, index=False)
    print(f"Feature-target correlations saved to {output_excel}")
    return df


def compute_feature_correlation_matrix(X, output_excel="../results/pearson_feature_correlation_matrix.xlsx"):
    """Compute Pearson correlation matrix among all features."""
    corr_matrix = X.corr(method="pearson")
    corr_matrix.to_excel(output_excel)
    print(f"Feature correlation matrix saved to {output_excel}")
    return corr_matrix


def plot_correlation_heatmap(corr_matrix,
                             output_png="../results/pearson_feature_correlation_matrix_heatmap.png"):
    """Heatmap of full feature correlation matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title("Pearson correlation matrix of input features")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to {output_png}")


def plot_top_features_heatmap(corr_matrix, pearson_df, top_n=15,
                              output_png="../results/pearson_top_features_correlation_heatmap.png"):
    """Heatmap of correlation among top N features."""
    top_features = pearson_df.head(top_n)["Feature"].values
    corr_sub = corr_matrix.loc[top_features, top_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_sub,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Pearson correlation among top correlated features", fontsize=14)
    ax.set_xticks(np.arange(len(top_features)) + 0.5)
    ax.set_yticks(np.arange(len(top_features)) + 0.5)
    ax.set_xticklabels(top_features, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(top_features, fontsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Top features heatmap saved to {output_png}")


def main():
    X, y = load_data()

    # Feature vs target
    pearson_df = compute_pearson_feature_target(X, y)

    # Feature vs feature
    corr_matrix = compute_feature_correlation_matrix(X)
    plot_correlation_heatmap(corr_matrix)
    plot_top_features_heatmap(corr_matrix, pearson_df)

    print("Pearson correlation analysis completed.")


if __name__ == "__main__":
    main()
