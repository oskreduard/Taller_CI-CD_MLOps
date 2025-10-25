import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Tracking local (carpeta mlruns en este repo)
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("ci-cd-mlflow-local")

# Datos
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

with mlflow.start_run():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    # Log de métrica y modelo
    mlflow.log_metric("mse", float(mse))
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ MSE: {mse:.4f}")
