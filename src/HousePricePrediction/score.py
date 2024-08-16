# score.py

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from HousePricePrediction.logger import get_logger
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import mlflow
import mlflow.sklearn

def load_housing_data(processed_data_path):
    """
    Load housing data from a specified directory.

    Parameters
    ----------
    processed_data_path : str
        The path to the directory containing the processed housing data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the housing data.
    """
    csv_path = os.path.join(processed_data_path, "housing.csv")
    return pd.read_csv(csv_path)


def main(args):
    """
    Main function to execute the scoring process.

    This function loads the housing data, splits it into training and testing
    sets, preprocesses the data, loads pre-trained models, makes predictions,
    and evaluates the models using RMSE and MAE.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments passed to the script.
    """
    logger = get_logger("score", "score.log")
    housing = load_housing_data(args.data_path)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        """
        Calculate the proportions of income categories in the data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data.

        Returns
        -------
        pd.Series
            A Series containing the proportions of income categories.
        """
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

    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(
        X, columns=housing_num.columns, index=housing.index
    )
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    lin_reg = joblib.load(os.path.join(args.model_path, "lin_reg.pkl"))
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("lin_rmse :", lin_rmse)
    logger.info(f"lin_rmse {lin_rmse}")
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    # print("lin_mae: ", lin_mae)
    logger.info(f"lin_mae, {lin_mae}")

    tree_reg = joblib.load(os.path.join(args.model_path, "tree_reg.pkl"))
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    # print("tree_rmse", tree_rmse)
    logger.info(f"tree_rmse, {tree_rmse}")

    forest_reg = joblib.load(os.path.join(args.model_path, "forest_reg.pkl"))
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    final_predictions = forest_reg.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(f"Forest RMSE: {final_rmse}")
    logger.info(f"Forest RMSE: {final_rmse}")

    return final_mse, tree_rmse, lin_mae
 

    

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
        help="Path to load trained model",
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
    final_mse, tree_rmse, lin_mae = main(args)
    with mlflow.start_run():
        mlflow.log_metric("Forest RMSE", final_mse[0])
        mlflow.log_metric("tree_rmse", tree_rmse[0])
        mlflow.log_metric("lin_mae", lin_mae[0])
