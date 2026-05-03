import numpy as np
import pandas as pd
import joblib


# Element properties for Ef model feature computation
ELEMENT_PROPERTIES = {
    # A-site elements
    "Y":  {"radius": 101.9, "mass": 88.9,  "vec": 3,  "electro": 1.22, "melt": 1799},
    "La": {"radius": 116.0, "mass": 138.9, "vec": 3,  "electro": 1.10, "melt": 1193},
    "Ce": {"radius": 114.3, "mass": 140.1, "vec": 4,  "electro": 1.12, "melt": 1071},
    "Pr": {"radius": 112.6, "mass": 140.9, "vec": 5,  "electro": 1.13, "melt": 1204},
    "Nd": {"radius": 110.9, "mass": 144.2, "vec": 6,  "electro": 1.14, "melt": 1294},
    "Sm": {"radius": 107.9, "mass": 150.3, "vec": 8,  "electro": 1.17, "melt": 1345},
    "Eu": {"radius": 103.3, "mass": 152.0, "vec": 9,  "electro": 1.20, "melt": 1095},
    "Gd": {"radius": 105.3, "mass": 157.3, "vec": 10, "electro": 1.20, "melt": 1586},
    "Tb": {"radius": 104.0, "mass": 158.9, "vec": 11, "electro": 1.22, "melt": 1629},
    "Dy": {"radius": 102.7, "mass": 162.5, "vec": 12, "electro": 1.23, "melt": 1685},
    "Ho": {"radius": 101.5, "mass": 164.9, "vec": 13, "electro": 1.24, "melt": 1747},
    "Er": {"radius": 100.2, "mass": 167.3, "vec": 14, "electro": 1.24, "melt": 1770},
    "Yb": {"radius": 98.5,  "mass": 173.0, "vec": 16, "electro": 1.10, "melt": 1092},
    "Lu": {"radius": 97.8,  "mass": 175.0, "vec": 3,  "electro": 1.27, "melt": 1936},
    # B-site elements
    "Ti": {"radius": 60.5, "mass": 47.9,  "vec": 4, "electro": 1.54, "melt": 1941},
    "Zr": {"radius": 72.0, "mass": 91.0,  "vec": 4, "electro": 1.33, "melt": 2128},
    "Hf": {"radius": 71.0, "mass": 178.0, "vec": 4, "electro": 1.30, "melt": 2506},
    "Sn": {"radius": 69.0, "mass": 118.7, "vec": 4, "electro": 1.96, "melt": 505},
}

A_ELEMENTS = ["Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"]
B_ELEMENTS = ["Ti", "Zr", "Hf", "Sn"]

# Composition column names
COLS_A = [f"X_{e}" for e in A_ELEMENTS]
COLS_B = [f"X_{e}" for e in B_ELEMENTS]

# Antisite marker column names (in input file)
COLS_A2 = [f"{e}_2" for e in A_ELEMENTS]
COLS_B2 = [f"{e}_2" for e in B_ELEMENTS]

# Mapping from antisite marker column to element name
A2_TO_ELEM = dict(zip(COLS_A2, A_ELEMENTS))
B2_TO_ELEM = dict(zip(COLS_B2, B_ELEMENTS))


def compute_ef_features(df):
    """
    Compute all 62 features for the Ef model from composition + antisite markers.

    Input columns needed:
        - X_Y ... X_Sn (18 composition columns)
        - Y_2 ... Sn_2 (18 antisite marker columns)

    Output: DataFrame with 62 columns matching Ef model feature order.
    """
    n = len(df)

    # Get property arrays
    a_radii = np.array([ELEMENT_PROPERTIES[e]["radius"] for e in A_ELEMENTS])
    a_masses = np.array([ELEMENT_PROPERTIES[e]["mass"] for e in A_ELEMENTS])
    a_vecs = np.array([ELEMENT_PROPERTIES[e]["vec"] for e in A_ELEMENTS])
    a_electros = np.array([ELEMENT_PROPERTIES[e]["electro"] for e in A_ELEMENTS])
    a_melts = np.array([ELEMENT_PROPERTIES[e]["melt"] for e in A_ELEMENTS])

    b_radii = np.array([ELEMENT_PROPERTIES[e]["radius"] for e in B_ELEMENTS])
    b_masses = np.array([ELEMENT_PROPERTIES[e]["mass"] for e in B_ELEMENTS])
    b_electros = np.array([ELEMENT_PROPERTIES[e]["electro"] for e in B_ELEMENTS])
    b_melts = np.array([ELEMENT_PROPERTIES[e]["melt"] for e in B_ELEMENTS])

    # Composition weights
    a_weights = df[COLS_A].values  # (n, 14)
    b_weights = df[COLS_B].values  # (n, 4)

    # --- Bulk physical features (weighted averages) ---
    a1_radius = a_weights @ a_radii
    b1_radius = b_weights @ b_radii
    ra1_rb1 = a1_radius / b1_radius

    a1_mass = a_weights @ a_masses
    b1_mass = b_weights @ b_masses
    a1_b1_mass = a1_mass / b1_mass

    a_vec = a_weights @ a_vecs
    a_electro = a_weights @ a_electros
    b_electro = b_weights @ b_electros

    a_melt = a_weights @ a_melts
    b_melt = b_weights @ b_melts
    a_b_melt = a_melt / b_melt
    a_minus_b_melt = a_melt - b_melt

    # --- Antisite markers ---
    anti_a = df[COLS_A2].values  # (n, 14) - one-hot for antisite A element
    anti_b = df[COLS_B2].values  # (n, 4) - one-hot for antisite B element

    # --- Antisite physical features (single element properties) ---
    a_radius_anti = anti_a @ a_radii
    b_radius_anti = anti_b @ b_radii
    ra_rb_anti = np.where(b_radius_anti > 0, a_radius_anti / b_radius_anti, 0)

    a_vec_anti = anti_a @ a_vecs
    a_electro_anti = anti_a @ a_electros
    b_electro_anti = anti_b @ b_electros

    a_mass_anti = anti_a @ a_masses
    b_mass_anti = anti_b @ b_masses
    a_b_mass_anti = np.where(b_mass_anti > 0, a_mass_anti / b_mass_anti, 0)

    a_melt_anti = anti_a @ a_melts
    b_melt_anti = anti_b @ b_melts
    a_b_melt_anti = np.where(b_melt_anti > 0, a_melt_anti / b_melt_anti, 0)
    a_minus_b_melt_anti = a_melt_anti - b_melt_anti

    # --- Build output DataFrame ---
    result = pd.DataFrame({
        # Composition (18)
        **{col: df[col].values for col in COLS_A + COLS_B},
        # Bulk physical features (13)
        "A1 Radius": a1_radius,
        "B1 Radius": b1_radius,
        "RA1/RB1": ra1_rb1,
        "A1 Mass": a1_mass,
        "B1 Mass": b1_mass,
        "A1/B1 Mass": a1_b1_mass,
        "A VEC": a_vec,
        "A electro": a_electro,
        "B electro": b_electro,
        "A melt": a_melt,
        "B melt": b_melt,
        "A/B melt": a_b_melt,
        "A-B melt": a_minus_b_melt,
        # Antisite markers (18)
        "X_Y_anti": anti_a[:, 0], "X_La_anti": anti_a[:, 1],
        "X_Ce_anti": anti_a[:, 2], "X_Pr_anti": anti_a[:, 3],
        "X_Nd_anti": anti_a[:, 4], "X_Sm_anti": anti_a[:, 5],
        "X_Eu_anti": anti_a[:, 6], "X_Gd_anti": anti_a[:, 7],
        "X_Tb_anti": anti_a[:, 8], "X_Dy_anti": anti_a[:, 9],
        "X_Ho_anti": anti_a[:, 10], "X_Er_anti": anti_a[:, 11],
        "X_Yb_anti": anti_a[:, 12], "X_Lu_anti": anti_a[:, 13],
        "X_Ti_anti": anti_b[:, 0], "X_Zr_anti": anti_b[:, 1],
        "X_Hf_anti": anti_b[:, 2], "X_Sn_anti": anti_b[:, 3],
        # Antisite physical features (13)
        "A Radius_anti": a_radius_anti,
        "B Radius_anti": b_radius_anti,
        "RA/RB_anti": ra_rb_anti,
        "A VEC_anti": a_vec_anti,
        "A electro_anti": a_electro_anti,
        "B electro_anti": b_electro_anti,
        "A Mass_anti": a_mass_anti,
        "B Mass_anti": b_mass_anti,
        "A/B Mass_anti": a_b_mass_anti,
        "A melt_anti": a_melt_anti,
        "B melt_anti": b_melt_anti,
        "A/B melt_anti": a_b_melt_anti,
        "A-B melt_anti": a_minus_b_melt_anti,
    })

    return result


def main():
    input_file = "data/pyrochlore_antisite_predictions_feature.xlsx"
    output_file = "results/pyrochlore_predictions_with_antisite_energy.xlsx"

    print("Loading antisite pair data...")
    df = pd.read_excel(input_file)
    print(f"Input shape: {df.shape}")

    print("Computing features for Ef model...")
    features = compute_ef_features(df)
    print(f"Feature matrix shape: {features.shape}")

    print("Loading model and predicting...")
    model = joblib.load("../../Cation_antisite_formation_energy_prediction/models/rf_model.joblib")
    scaler = joblib.load("../../Cation_antisite_formation_energy_prediction/models/scaler.joblib")

    X_scaled = scaler.transform(features.values)
    predictions = model.predict(X_scaled)

    # Output all 62 features + prediction (same format as Ef_train_data.xlsx)
    features["Predicted_Formation_Energy"] = predictions

    # Save results
    features.to_excel(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"\nFormation energy statistics:")
    print(pd.Series(predictions).describe())


if __name__ == "__main__":
    main()
