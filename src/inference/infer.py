import os
import time
import logging
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Setup logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# MLflow setup
mlflow.set_tracking_uri("file:/app/mlruns")
mlflow.set_experiment("Sentiment Classifier Inference")

def run_inference():
    start_time = time.time()
    run = None

    try:
        run = mlflow.start_run(run_name="Sentiment Inference")

        # Load inference data
        df = pd.read_parquet("/app/data/processed/processed_test.parquet")
        logging.info("Loaded inference data. Shape: %s", df.shape)
        mlflow.log_param("inference_dataset_size", df.shape[0])

        if "review" not in df.columns:
            raise ValueError("Input CSV must contain a 'review' column.")

        reviews = df["cleaned"].astype(str)

        # Load model and vectorizer
        model_path = "/app/outputs/models/logreg_model.pkl"
        vec_path = "/app/outputs/models/tfidf.pkl"

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            raise FileNotFoundError("Model or vectorizer not found.")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)

        # Vectorize the input
        X = vectorizer.transform(reviews)

        # Predict
        predictions = model.predict(X)
        proba = model.predict_proba(X)[:, 1]

        os.makedirs("/app/outputs/predictions", exist_ok=True)
        if "sentiment" in df.columns:
            predicted_sentiment = pd.Series(predictions).map({1: "positive", 0: "negative"})
            label_df = pd.DataFrame({
                "true_sentiment": df["sentiment"],
                "predicted_sentiment": predicted_sentiment
            })
            label_path = "/app/outputs/predictions/predictions.csv"
            label_df.to_csv(label_path, index=False)
            mlflow.log_artifact(label_path)

        # If true sentiment exists, evaluate
        if "sentiment" in df.columns:
            y_true = df["sentiment"]
            acc = (y_true == predictions).mean()
            roc_auc = roc_auc_score(y_true, proba)
            report = classification_report(y_true, predictions, target_names=["negative", "positive"])

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc_auc)

            report_path = "/app/outputs/predictions/classification_report.txt"
            with open(report_path, "w") as f:
                f.write("Accuracy: {:.2f}\n".format(acc))
                f.write(report)
            mlflow.log_artifact(report_path)

            # Confusion matrix
            cm = confusion_matrix(y_true, predictions)
            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar(cax)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks([0, 1], ["negative", "positive"])
            plt.yticks([0, 1], ["negative", "positive"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

            plt.tight_layout()
            cm_path = "/app/outputs/predictions/confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, proba)
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            roc_path = "/app/outputs/predictions/roc_auc_curve.png"
            plt.savefig(roc_path)
            plt.close()
            mlflow.log_artifact(roc_path)

        duration = time.time() - start_time
        mlflow.log_metric("inference_duration_sec", duration)
        logging.info("Inference completed in %.2fs", duration)

    except Exception as e:
        logging.error("Inference failed: %s", e)
        if run:
            mlflow.end_run(status="FAILED")

    finally:
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    run_inference()