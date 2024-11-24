import pandas as pd
import eda
import preprocessing
import feature_selection
import feature_engineering
import clustering
import visualization

def main():
    # Load dataset
    df = pd.read_csv('dataset.csv')

    # Step 1: Exploratory Data Analysis (EDA)
    print("Performing Exploratory Data Analysis...")
    eda.plot_heatmap(df)
    eda.plot_crimes_per_month(df)

    # Step 2: Preprocessing
    print("Preprocessing data...")
    df = preprocessing.reset_index(df)
    df = preprocessing.convert_column_types(df, ['some_column'], dtype='int')
    print(preprocessing.check_missing_values(df))
    print(preprocessing.check_unique_values(df))

    # Step 3: Feature Selection
    print("Selecting features...")
    entropies = feature_selection.calculate_entropies(df, df.columns)
    print("Entropies:", entropies)
    df = feature_selection.drop_high_entropy_features(df, ['column_to_drop'])

    # Step 4: Feature Engineering
    print("Engineering features...")
    data_train, data_test, target_train, target_test = feature_engineering.split_data(df, target='target_column')
    df = feature_engineering.one_hot_encode(df, columns=['categorical_column'])

    # Step 5: Clustering
    print("Applying clustering algorithms...")
    kmodes_clusters = clustering.apply_kmodes(df, n_clusters=3)
    hierarchical_clusters = clustering.hierarchical_clustering(df, threshold=0.1)
    spectral_clusters = clustering.spectral_clustering(df, n_clusters=3)

    # Step 6: Visualization
    print("Visualizing clustering results...")
    features_to_evaluate = ['feature1', 'feature2']
    visualization.plot_kmodes_clusters(df, kmodes_clusters, features_to_evaluate)
    visualization.plot_spectral_clusters(df, spectral_clusters, features_to_evaluate)

    print("Process completed.")

if __name__ == "__main__":
    main()
