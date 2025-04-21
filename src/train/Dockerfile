# Base image
FROM python:3.10-slim
 
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
 
# Set working directory in container
WORKDIR /app
 
# Copy training script
COPY ./src/train/train.py /app/train.py
 
# Copy requirements
COPY ./requirements.txt /app/requirements.txt
 
# Copy data directory (raw data should already be there; models and outputs are .gitignored)
COPY ./data /app/data
 
# Create output directories for storing results (models, predictions, metrics)
RUN mkdir -p /app/outputs/models /app/outputs/predictions /app/outputs/figures
 
# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
 
# Set environment variables for MLflow tracking
ENV MLFLOW_TRACKING_URI=file:/app/outputs/mlruns
ENV GIT_PYTHON_REFRESH=quiet
 
# Command to run training
CMD ["python", "train.py"]