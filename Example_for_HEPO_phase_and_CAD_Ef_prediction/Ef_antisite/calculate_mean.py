import pandas as pd


def main():
    print("Loading data...")
    df = pd.read_excel("results/pyrochlore_predictions_with_antisite_energy.xlsx")

    # Define group columns (first 18 columns)
    group_columns = df.columns[:18].tolist()

    # Round composition columns to avoid floating-point groupby issues
    df[group_columns] = df[group_columns].round(10)

    # Define columns to calculate mean
    mean_columns = ["RA1/RB1", "Predicted_Formation_Energy"]

    # Group by first 18 columns and calculate mean
    print("Aggregating data...")
    result = df.groupby(group_columns)[mean_columns].mean().reset_index()

    print(f"Aggregated data rows: {len(result)}")

    # Save results
    output_file = "results/pyrochlore_aggregated_Ef_antisite_results.xlsx"
    result.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Preview
    print("\nResults preview:")
    print(result.head())

    print("\nNon-null counts per column:")
    print(result.count())


if __name__ == "__main__":
    main()
