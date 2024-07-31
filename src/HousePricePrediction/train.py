import argparse
import os

import joblib
import numpy as np
import pandas as pd
from HousePricePrediction.logger import get_logger
from scipy.stats import randint

# from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):  # no *args or **kargs
        pass

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]


def load_housing_data(processed_data_path):
    """
    Load housing data from the specified directory.

    Parameters
    ----------
    processed_data_path : str
        Directory path to load the processed housing data from.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the housing data.
    """
    csv_path = os.path.join(processed_data_path, "housing.csv")
    return pd.read_csv(csv_path)


def main(args):
    """
    Main function to train machine learning models on housing data.

    This function handles the data loading, processing, training of
    different models, hyperparameter tuning, and saving the trained models.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments passed to the script.

    Returns
    -------
    None
    """
    log_file = args.log_path if args.log_path.lower() != "none" else None
    logger = get_logger("train", log_file, args.log_level, args.no_console_log)

    housing = load_housing_data(args.data_path)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )

    # Added Pipeline
    attr_adder = CombinedAttributesAdder()
    housing_extra_attribs = attr_adder.transform(housing_tr.values)

    cols = list(housing.columns)
    cols.extend(
        [
            "rooms_per_household",
            "population_per_household",
            "bedroom_per_household",
        ]
    )
    housing_tr = pd.DataFrame(housing_extra_attribs, columns=cols)

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # housing_predictions = lin_reg.predict(housing_prepared)
    # lin_mse = mean_squared_error(housing_labels, housing_predictions)
    # lin_rmse = np.sqrt(lin_mse)
    # lin_rmse

    # lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    # lin_mae

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    # housing_predictions = tree_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_labels, housing_predictions)
    # tree_rmse = np.sqrt(tree_mse)
    # tree_rmse

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    final_model = grid_search.best_estimator_

    os.makedirs(args.model_path, exist_ok=True)
    joblib.dump(lin_reg, os.path.join(args.model_path, "lin_reg.pkl"))
    joblib.dump(tree_reg, os.path.join(args.model_path, "tree_reg.pkl"))
    joblib.dump(
        final_model,
        os.path.join(args.model_path, "forest_reg.pkl"),
    )

    logger.info(f"Model trained and saved to {args.model_path}")

    return lin_reg, tree_reg, forest_reg
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed",
        help="Path to processed data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="artifacts",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Specify log level",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="None",
        help="Use a log file or not, give path",
    )
    parser.add_argument(
        "--no_console_log",
        action="store_true",
        help="Not to write logs to console",
    )
    args = parser.parse_args()
    lin_reg, tree_reg, forest_reg = main(args)

    with mlflow.start_run():
        mlflow.sklearn.log_model(lin_reg, "model/lin_reg")
        mlflow.sklearn.log_model(tree_reg, "model/tree_reg")
        mlflow.sklearn.log_model(forest_reg, "model/forest_reg")
