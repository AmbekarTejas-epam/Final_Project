# import os
# import time
# import logging
# import joblib
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import mlflow
# import mlflow.sklearn

# """
# This script trains a Logistic Regression model for sentiment classification using scikit-learn.
# It includes:
# - Loading preprocessed data
# - TF-IDF vectorization
# - Model training and evaluation
# - Logging metrics, parameters, and model artifact using MLflow
# """

# # Setup logging
# log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# if logger.hasHandlers():
#     logger.handlers.clear()

# console_handler = logging.StreamHandler()
# console_handler.setFormatter(log_formatter)
# logger.addHandler(console_handler)

# # MLflow config
# mlflow.set_tracking_uri("file:/app/mlruns")
# mlflow.set_experiment("Sentiment Logistic Regression")


# def train(test_size=0.15):
#     start_time = time.time()
#     run = None

#     try:
#         # Load data
#         data_path = "/app/data/processed/processed_train.parquet"
#         df = pd.read_parquet(data_path)
#         logger.info("Data loaded from %s | Shape: %s", data_path, df.shape)

#         # Use lemmatized text
#         texts = df['cleaned']
#         labels = df['sentiment']

#         # Vectorization
#         vectorizer = TfidfVectorizer(min_df=25, max_df=0.85)
#         X = vectorizer.fit_transform(texts)
#         y = labels

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42, stratify=y
#         )

#         logger.info("Split: Train=%d, Test=%d", X_train.shape[0], X_test.shape[0])

#         model = LogisticRegression(random_state=42)

#         run = mlflow.start_run()
#         mlflow.log_param("test_size", test_size)
#         mlflow.log_param("vectorizer", "TfidfVectorizer")
#         mlflow.log_param("model_type", "LogisticRegression")

#         model.fit(X_train, y_train)
#         logger.info("Model trained.")

#         # Evaluation
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         logger.info("Test Accuracy: %.4f", acc)

#         mlflow.log_metric("accuracy", acc)

#         # Save model and vectorizer
#         os.makedirs("/app/outputs/models", exist_ok=True)
#         model_path = "/app/outputs/models/logreg_model.pkl"
#         vectorizer_path = "/app/outputs/models/tfidf.pkl"


#         joblib.dump(model, model_path)
#         joblib.dump(vectorizer, vectorizer_path)
#         logger.info("Model saved at %s", model_path)

#         mlflow.sklearn.log_model(model, "logreg_model")
#         mlflow.log_artifact(vectorizer_path)

#         duration = time.time() - start_time
#         mlflow.log_metric("training_duration_sec", duration)
#         logger.info("Training completed in %.2f seconds", duration)

#     except Exception as e:
#         logger.error("Training failed due to: %s", str(e))
#         if run:
#             mlflow.end_run(status="FAILED")
#     finally:
#         if mlflow.active_run():
#             mlflow.end_run(status="FINISHED")
#         logger.info("Run ended.")
#         logger.info("Total time taken: %.2f seconds", time.time() - start_time)


# if __name__ == "__main__":
#     train()
import os
import time
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import mlflow
import mlflow.sklearn

"""
This script trains a Logistic Regression model for sentiment classification using scikit-learn.
It includes:
- Loading preprocessed data
- TF-IDF vectorization
- Dimensionality Reduction using SVD
- Model training and evaluation
- Logging metrics, parameters, and model artifact using MLflow
"""

# Setup logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# MLflow config
mlflow.set_tracking_uri("file:/app/mlruns")
mlflow.set_experiment("Sentiment Logistic Regression")


def train(test_size=0.15):
    start_time = time.time()
    run = None

    try:
        # Load data
        data_path = "/app/data/processed/processed_train.parquet"
        df = pd.read_parquet(data_path)
        logger.info("Data loaded from %s | Shape: %s", data_path, df.shape)

        # Use lemmatized text
        texts = df['cleaned']
        labels = df['sentiment']

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(min_df=25, max_df=0.85)
        X_tfidf = vectorizer.fit_transform(texts)
        y = labels

        # Dimensionality Reduction using SVD
        svd_components = 130
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        X = svd.fit_transform(X_tfidf)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info("Split: Train=%d, Test=%d", X_train.shape[0], X_test.shape[0])

        # Model setup
        model = LogisticRegression(random_state=42)

        run = mlflow.start_run()
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("svd_components", svd_components)

        model.fit(X_train, y_train)
        logger.info("Model trained.")

        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("Test Accuracy: %.4f", acc)
        mlflow.log_metric("accuracy", acc)

        # Save model and transformers
        os.makedirs("/app/outputs/models", exist_ok=True)
        model_path = "/app/outputs/models/logreg_model.pkl"
        vectorizer_path = "/app/outputs/models/tfidf.pkl"
        svd_path = "/app/outputs/models/svd.pkl"

        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(svd, svd_path)

        logger.info("Model and transformers saved.")

        mlflow.sklearn.log_model(model, "logreg_model")
        mlflow.log_artifact(vectorizer_path)
        mlflow.log_artifact(svd_path)

        # Log training time
        duration = time.time() - start_time
        mlflow.log_metric("training_duration_sec", duration)
        logger.info("Training completed in %.2f seconds", duration)

    except Exception as e:
        logger.error("Training failed due to: %s", str(e))
        if run:
            mlflow.end_run(status="FAILED")
    finally:
        if mlflow.active_run():
            mlflow.end_run(status="FINISHED")
        logger.info("Run ended.")
        logger.info("Total time taken: %.2f seconds", time.time() - start_time)


if __name__ == "__main__":
    train()
