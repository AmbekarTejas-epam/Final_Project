# Base image
FROM python:3.10-slim
 
# Set working directory inside the container
WORKDIR /app
 
# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy inference script
COPY src/inference/infer.py /app/infer.py
 
# Set MLflow environment variables
ENV MLFLOW_TRACKING_URI=file:/app/outputs/mlruns
ENV GIT_PYTHON_REFRESH=quiet
 
# Create required output directories
RUN mkdir -p /app/outputs/predictions /app/mlruns
 
# Set default command
CMD ["python", "infer.py"]