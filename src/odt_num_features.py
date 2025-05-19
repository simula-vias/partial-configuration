import json
from common import load_data
from odt_labelled import prepare_perf_matrx, binarize

metadata = json.load(open("./data/metadata.json"))

for system in metadata.keys():
    (
        perf_matrix_initial,
        input_features,
        config_features,
        all_performances,
        input_preprocessor,
        config_preprocessor,
    ) = load_data(
        system=system, data_dir="./data", input_properties_type="tabular"
    )

    perf_matrix = prepare_perf_matrx(perf_matrix_initial, all_performances)

    input_feature_columns = input_features.columns
    feature_matrix = perf_matrix[list(input_feature_columns) + ["inputname"]].drop_duplicates()
    features = feature_matrix[input_feature_columns]
    inputnames = feature_matrix["inputname"]

    # Transform features
    X = input_preprocessor.fit_transform(features)
    feature_names = input_preprocessor.get_feature_names_out()

    # TODO This needs to be validated, n_bins is only used for real_cols
    # Maybe the input_preprocessor above is the better choice
    X_all = binarize(
        features,
        categorical_cols=[],
        integer_cols=features.columns,
        real_cols=[],
        n_bins=5,
    )
    print(system, X_all.shape, X.shape)
    print(X[0])