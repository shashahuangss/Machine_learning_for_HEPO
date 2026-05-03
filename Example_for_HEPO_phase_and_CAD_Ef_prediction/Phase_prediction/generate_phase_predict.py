import numpy as np
import pandas as pd


# Element properties from training data
ELEMENT_PROPERTIES = {
    # A-site elements (14 rare earths)
    "Y":  {"radius": 101.9, "mass": 88.9, "vec": 3,  "electro": 1.22},
    "La": {"radius": 116.0, "mass": 138.9, "vec": 3,  "electro": 1.10},
    "Ce": {"radius": 114.3, "mass": 140.1, "vec": 4,  "electro": 1.12},  # estimated
    "Pr": {"radius": 112.6, "mass": 140.9, "vec": 5,  "electro": 1.13},
    "Nd": {"radius": 110.9, "mass": 144.2, "vec": 6,  "electro": 1.14},
    "Sm": {"radius": 107.9, "mass": 150.3, "vec": 8,  "electro": 1.17},
    "Eu": {"radius": 103.3, "mass": 152.0, "vec": 9,  "electro": 1.20},
    "Gd": {"radius": 105.3, "mass": 157.3, "vec": 10, "electro": 1.20},
    "Tb": {"radius": 104.0, "mass": 158.9, "vec": 11, "electro": 1.22},
    "Dy": {"radius": 102.7, "mass": 162.5, "vec": 12, "electro": 1.23},
    "Ho": {"radius": 101.5, "mass": 164.9, "vec": 13, "electro": 1.24},
    "Er": {"radius": 100.2, "mass": 167.3, "vec": 14, "electro": 1.24},
    "Yb": {"radius": 98.5,  "mass": 173.0, "vec": 16, "electro": 1.10},
    "Lu": {"radius": 97.8,  "mass": 175.0, "vec": 3,  "electro": 1.27},
    # B-site elements (4)
    "Ti": {"radius": 60.5, "mass": 47.9,  "vec": 4, "electro": 1.54},
    "Zr": {"radius": 72.0, "mass": 91.0,  "vec": 4, "electro": 1.33},
    "Hf": {"radius": 71.0, "mass": 178.0,  "vec": 4, "electro": 1.30},
    "Sn": {"radius": 69.0, "mass": 118.7, "vec": 4, "electro": 1.96},
}

A_SITE_ELEMENTS = ["Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"]
B_SITE_ELEMENTS = ["Ti", "Zr", "Hf", "Sn"]


def generate_random_compositions(num_samples=1000):
    """Generate random A/B site compositions with equal weights."""
    def generate_a_site(num_samples):
        a_site = np.zeros((num_samples, 14), dtype=float)
        for i in range(num_samples):
            num_nonzero = np.random.randint(2, 7)
            nonzero_indices = np.random.choice(14, num_nonzero, replace=False)
            weights = np.random.dirichlet(np.ones(num_nonzero))
            a_site[i, nonzero_indices] = weights
        return a_site

    def generate_b_site(num_samples):
        b_site = np.zeros((num_samples, 4), dtype=float)
        for i in range(num_samples):
            buckets = np.random.choice(4, 16)
            for b in buckets:
                b_site[i, b] += 1
        return b_site / 16

    return generate_a_site(num_samples), generate_b_site(num_samples)


def compute_weighted_avg(values, weights):
    """Compute weighted average."""
    return np.sum(values * weights)


def compute_delta(values, weights, avg):
    """
    Compute delta (coefficient of variation).
    delta = sqrt(sum(w_i * (v_i - avg)^2)) / avg
    """
    std = np.sqrt(np.sum(weights * (values - avg) ** 2))
    return std / avg if avg > 0 else 0.0


def compute_features(a_site, b_site):
    """
    Compute all 36 features from A/B site compositions.

    Formulas:
    - Weighted average: sum(x_i * p_i)
    - Delta (CV): sqrt(sum(x_i * (p_i - avg)^2)) / avg
    - size_disorder: sqrt(delta_R_A^2 + delta_R_B^2)
    - mass_disorder: sqrt(delta_mass_A^2 + delta_mass_B^2)
    """
    n_samples = a_site.shape[0]

    # Initialize arrays
    a_radius = np.zeros(n_samples)
    a_delta_r = np.zeros(n_samples)
    b_radius = np.zeros(n_samples)
    b_delta_r = np.zeros(n_samples)
    size_disorder = np.zeros(n_samples)
    ra_rb = np.zeros(n_samples)

    a_mass = np.zeros(n_samples)
    a_delta_mass = np.zeros(n_samples)
    b_mass = np.zeros(n_samples)
    b_delta_mass = np.zeros(n_samples)
    mass_ratio = np.zeros(n_samples)
    mass_disorder = np.zeros(n_samples)

    a_vec = np.zeros(n_samples)
    a_delta_vec = np.zeros(n_samples)
    a_electro = np.zeros(n_samples)
    a_delta_electro = np.zeros(n_samples)
    b_electro = np.zeros(n_samples)
    b_delta_electro = np.zeros(n_samples)

    for i in range(n_samples):
        # A-site
        a_weights = a_site[i]
        a_radii = np.array([ELEMENT_PROPERTIES[e]["radius"] for e in A_SITE_ELEMENTS])
        a_masses = np.array([ELEMENT_PROPERTIES[e]["mass"] for e in A_SITE_ELEMENTS])
        a_vecs = np.array([ELEMENT_PROPERTIES[e]["vec"] for e in A_SITE_ELEMENTS])
        a_electros = np.array([ELEMENT_PROPERTIES[e]["electro"] for e in A_SITE_ELEMENTS])

        a_radius[i] = compute_weighted_avg(a_radii, a_weights)
        a_delta_r[i] = compute_delta(a_radii, a_weights, a_radius[i])
        a_mass[i] = compute_weighted_avg(a_masses, a_weights)
        a_delta_mass[i] = compute_delta(a_masses, a_weights, a_mass[i])
        a_vec[i] = compute_weighted_avg(a_vecs, a_weights)
        a_delta_vec[i] = compute_delta(a_vecs, a_weights, a_vec[i])
        a_electro[i] = compute_weighted_avg(a_electros, a_weights)
        a_delta_electro[i] = compute_delta(a_electros, a_weights, a_electro[i])

        # B-site
        b_weights = b_site[i]
        b_radii = np.array([ELEMENT_PROPERTIES[e]["radius"] for e in B_SITE_ELEMENTS])
        b_masses = np.array([ELEMENT_PROPERTIES[e]["mass"] for e in B_SITE_ELEMENTS])
        b_electros = np.array([ELEMENT_PROPERTIES[e]["electro"] for e in B_SITE_ELEMENTS])

        b_radius[i] = compute_weighted_avg(b_radii, b_weights)
        b_delta_r[i] = compute_delta(b_radii, b_weights, b_radius[i])
        b_mass[i] = compute_weighted_avg(b_masses, b_weights)
        b_delta_mass[i] = compute_delta(b_masses, b_weights, b_mass[i])
        b_electro[i] = compute_weighted_avg(b_electros, b_weights)
        b_delta_electro[i] = compute_delta(b_electros, b_weights, b_electro[i])

        # Derived features
        ra_rb[i] = a_radius[i] / b_radius[i] if b_radius[i] > 0 else 0.0
        size_disorder[i] = np.sqrt(a_delta_r[i] ** 2 + b_delta_r[i] ** 2)
        mass_ratio[i] = a_mass[i] / b_mass[i] if b_mass[i] > 0 else 0.0
        mass_disorder[i] = np.sqrt(a_delta_mass[i] ** 2 + b_delta_mass[i] ** 2)

    # Build DataFrame
    column_names = [
        "X_Y", "X_La", "X_Ce", "X_Pr", "X_Nd", "X_Sm", "X_Eu", "X_Gd",
        "X_Tb", "X_Dy", "X_Ho", "X_Er", "X_Yb", "X_Lu",
        "X_Ti", "X_Zr", "X_Hf", "X_Sn",
        "A1 Radius", "A1 delta R", "B1 Radius", "B1 delta R",
        "size disorder", "RA1/RB1",
        "mass A", "delta mass A", "mass B", "delta mass B",
        "mass A/B", "mass disorder",
        "A VEC", "delta A VEC", "A electro", "delta A electro",
        "B electro", "delta B electro"
    ]

    features = np.column_stack([
        a_site, b_site,
        a_radius, a_delta_r, b_radius, b_delta_r,
        size_disorder, ra_rb,
        a_mass, a_delta_mass, b_mass, b_delta_mass,
        mass_ratio, mass_disorder,
        a_vec, a_delta_vec, a_electro, a_delta_electro,
        b_electro, b_delta_electro
    ])

    return pd.DataFrame(features, columns=column_names)


def main():
    print("Generating random compositions...")
    num_samples = 10000
    a_site, b_site = generate_random_compositions(num_samples)

    print("Computing derived features...")
    features_df = compute_features(a_site, b_site)

    # Validation
    print("\nA-site sum (first 5 rows):")
    print(features_df.iloc[:, :14].sum(axis=1).head())

    print("\nB-site sum (first 5 rows):")
    print(features_df.iloc[:, 14:18].sum(axis=1).head())

    print("\nFeature statistics:")
    print(features_df.describe())

    # Save
    output_file = "data/random_pyrochlore_features_phase.xlsx"
    features_df.to_excel(output_file, index=False)
    print(f"\nGenerated {num_samples} samples saved to '{output_file}'")


if __name__ == "__main__":
    main()
