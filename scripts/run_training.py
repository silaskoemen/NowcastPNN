import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import logging
import json
from pathlib import Path

from nowcastpnn import data, models, train, evaluate
from nowcastpnn.utils.train_utils import set_seeds, mlflow_log_metrics

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_experiment(cfg: DictConfig):
    """
    Runs a training experiment using a Hydra configuration, with two-tiered persistence.
    1. Logs detailed results to MLflow for dynamic tracking.
    2. Saves a final, auditable summary JSON to the outputs directory.
    """
    set_seeds(cfg.seed)
    # Hydra automatically creates a unique output directory for each run.
    # This is where we will save our auditable summary.
    # Could also change naming of dir or stop entirely bc saved to MLFlow!
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Run output will be saved to: {output_dir}")

    mlflow.set_tracking_uri(train.MLFLOW_TRACKING_URI)

    # --- MLflow Setup ---
    mlflow.set_experiment(cfg.project_info.experiment_name)
    with mlflow.start_run(run_name=cfg.project_info.run_name) as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")
        # Currently, entire dict saved of config instead of individual params, could fix cast to dict for log_params
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        mlflow.log_param("hydra_output_path", str(output_dir))

        # --- Data Loading ---
        dataset = data.get_dataset_config(cfg)
        train_loader, val_loader, test_loader = data.get_loaders_from_dataset(dataset, cfg)
        logging.info("Loaded data.")

        # --- Model Initialization ---
        model = models.get_model(cfg) # Using a factory function is cleaner
        early_stopper = train.EarlyStopper(patience=cfg.training.patience)
        logging.info("Loaded model and early stopper.")

        # --- Training ---
        best_model, history = train.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stopper=early_stopper,
            loss_fct=cfg.training.loss_fct,
            num_epochs=cfg.training.epochs,
            learning_rate=cfg.training.learning_rate,
            device=cfg.training.device,
            dow=cfg.data.use_dow,
            return_history=cfg.training.return_history
        )

        # Log metrics to MLflow
        for key, values in history.items():
            for i, value in enumerate(values):
                mlflow.log_metric(key, value, step=i)

        # --- Evaluation ---
        final_metrics = {}
        if best_model and test_loader:
            final_metrics = evaluate.calculate_metrics(
                model=best_model,
                test_loader=test_loader,
                random_split=cfg.data.random_split,
                dow=cfg.data.use_dow
            )
            mlflow_log_metrics(final_metrics)
            logging.info(f"Final Metrics: {final_metrics}")

        # --- Save to MLFlow ---
        if best_model:
            mlflow.pytorch.log_model(best_model)  # type: ignore

if __name__ == "__main__":
    run_experiment()
