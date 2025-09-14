import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import os
import torch
import json
import pandas as pd
from pathlib import Path

from nowcastpnn import data, models, train

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_experiment(cfg: DictConfig):
    """
    Runs a training experiment using a Hydra configuration, with two-tiered persistence.
    1. Logs detailed results to MLflow for dynamic tracking.
    2. Saves a final, auditable summary JSON to the outputs directory.
    """
    # Hydra automatically creates a unique output directory for each run.
    # This is where we will save our auditable summary.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"Run output will be saved to: {output_dir}")

    # --- MLflow Setup ---
    mlflow.set_experiment(cfg.project_info.experiment_name)
    with mlflow.start_run(run_name=cfg.project_info.run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        mlflow.log_param("hydra_output_path", str(output_dir))

        # --- Data Loading ---
        train_loader, val_loader, test_loader = data.get_dataset(**cfg.data, dow=cfg.model.use_dow)

        # --- Model Initialization ---
        model = models.get_model(cfg) # Using a factory function is cleaner

        # --- Training ---
        early_stopper = train.EarlyStopper(patience=cfg.training.patience)
        best_model, history = train.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stopper=early_stopper,
            loss_fct=cfg.model.loss_fct,
        )

        # Log metrics to MLflow
        for key, values in history.items():
            for i, value in enumerate(values):
                mlflow.log_metric(key, value, step=i)

        # --- Evaluation ---
        final_metrics = {}
        if best_model and test_loader:
            predictions, targets = evaluate.predict(best_model, test_loader)
            final_metrics = evaluate.get_all_metrics(predictions, targets)
            mlflow.log_metrics(final_metrics)
            print(f"Final Metrics: {final_metrics}")

        # --- Tier 1 Persistence: MLflow Artifacts ---
        if best_model:
            mlflow.pytorch.log_model(best_model, "model")

        # --- Tier 2 Persistence: Auditable JSON Summary ---
        summary = {
            "mlflow_run_id": run_id,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "results": {
                "best_val_loss": early_stopper.best_loss,
                "final_metrics": final_metrics
            }
        }

        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Auditable summary saved to: {summary_path}")

if __name__ == "__main__":
    run_experiment()
