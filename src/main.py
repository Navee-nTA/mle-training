import mlflow
import argparse
from HousePricePrediction import ingest_data
from HousePricePrediction import train
from HousePricePrediction import score


# Create nested runs
experiment_id = mlflow.create_experiment("experiment1")
with mlflow.start_run(
    run_name="PARENT_RUN",
    experiment_id=experiment_id,
) as parent_run:
    with mlflow.start_run(
        run_name="ingest_data",
        experiment_id=experiment_id,
        nested=True,
    ) as child_run:
        params = {  
                    "raw_data_path": "data/raw",
                    "processed_data_path": "data/processed",
                    "log_level": "INFO",
                    "log_path": "None",
                    "no_console_log": True
                }
        args = argparse.Namespace(**params)

        ingest_data.main(args)
        mlflow.log_artifact(args.processed_data_path)

    with mlflow.start_run(
        run_name="train",
        experiment_id=experiment_id,
        nested=True,
    ) as child_run:
        params = {
                    "data_path": "data/processed",
                    "model_path": "artifacts",
                    "log_level": "INFO",
                    "log_path": "None",
                    "no_console_log": True
                }

# Create a Namespace object with these parameters
        args = argparse.Namespace(**params)
        lin_reg, tree_reg, forest_reg = train.main(args)
        mlflow.sklearn.log_model(lin_reg, "model/lin_reg")
        mlflow.sklearn.log_model(tree_reg, "model/tree_reg")
        mlflow.sklearn.log_model(forest_reg, "model/forest_reg")

    with mlflow.start_run(
        run_name="score",
        experiment_id=experiment_id,
        nested=True,
    ) as child_run:
        params = {
                    "data_path": "data/processed",
                    "model_path": "artifacts",
                    "log_level": "INFO",
                    "log_path": "None",
                    "no_console_log": True
                }

# Create an argparse.Namespace object with these parameters
        args = argparse.Namespace(**params)
        final_mse, tree_rmse, lin_mae = score.main(args)
        mlflow.log_metric("Forest RMSE", final_mse)
        mlflow.log_metric("tree_rmse", tree_rmse)
        mlflow.log_metric("lin_mae", lin_mae)

