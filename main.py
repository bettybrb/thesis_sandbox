

# ============================================================
# MAIN (pseudocode usage)
# ============================================================

def main():
    # 1) Preprocessing: build Original/ directory
    preprocess_all_subjects(gdf_dir=Path("PATH_TO_GDF_FILES"))

    # 2) Generation: build Generated/ directory
    generate_all_subjects(num_synthetic_per_class=200)  # example number

    # 3) Classification: compare Original-trained vs Generated-trained
    classification_experiment()


if __name__ == "__main__":
    main()