import sys
import mlflow
import pandas as pd

THRESHOLD = 3000.0  # Umbral de la guía

mlflow.set_tracking_uri("mlruns")
experiment = mlflow.get_experiment_by_name("ci-cd-mlflow-local")
if experiment is None:
    print("❌ No existe el experimento 'ci-cd-mlflow-local'.")
    sys.exit(1)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)

if runs.empty or "metrics.mse" not in runs.columns:
    print("❌ No se encontró métrica 'mse' en el último run.")
    sys.exit(1)

mse = float(runs.iloc[0]["metrics.mse"])
print(f"🔎 Validando MSE del último run: {mse:.4f}")

if mse > THRESHOLD:
    print(f"❌ MSE {mse:.4f} > {THRESHOLD} (falla la validación).")
    sys.exit(1)

print("✅ MSE aceptable. Modelo listo para promoción.")
