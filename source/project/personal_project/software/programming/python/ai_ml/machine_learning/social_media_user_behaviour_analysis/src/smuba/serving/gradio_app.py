"""Gradio demo UI - loads the current Production-stage model straight from
the local MLflow registry (no separate model file to keep in sync).
Flagged predictions still land in flagged/, same as your original app.
"""
import gradio as gr
import mlflow
import pandas as pd

from smuba.config import load_config

cfg = load_config()
mlflow.set_tracking_uri(cfg["pipeline"]["mlflow_tracking_uri"])
MODEL_NAME = cfg["pipeline"]["registered_model_name"]
FEATURES = cfg["data"]["selected_features"]

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")


def predict(*values):
    row = pd.DataFrame([dict(zip(FEATURES, values))])
    prediction = model.predict(row)
    return str(prediction[0])


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label=feature) for feature in FEATURES],
    outputs=gr.Textbox(label="Prediction"),
    title="Social Media User Behaviour Predictor",
    description="Serves the current 'Production'-stage model from the local MLflow registry.",
)

if __name__ == "__main__":
    demo.launch()
