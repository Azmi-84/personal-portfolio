MLflow (`mlflow.db` + `mlruns/`) is the actual model registry - it tracks
every version, its metrics, and which one is staged `Production`. This
folder is only for the optional, occasional case where you want a plain
`.pkl` export of the current Production model for use outside MLflow:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/smuba-model/Production")
# then export however you need, e.g. joblib.dump(model._model_impl, "models/export.pkl")
```
