
# Sentiment Analysis Project: Binary Classification of Movie Reviews

## Data Science Part

###  Introduction
This project aims to build a **binary sentiment classification model** to analyze movie reviews. The objective is to predict whether a review expresses a **positive** or **negative** sentiment. We utilized a dataset of **50,000 movie reviews**, which are split between training and testing. The project involves data preprocessing, EDA, feature engineering(tokenization, stop-words filtering, stemming, lemmatization and vectorization), model training, and evaluation.

---

###  Tools and Libraries Used

- **Platform**: VS code and Google Colab (for prototyping and EDA)
- **Languages & Libraries**:
  - `Python` (Core language)
  - `Pandas`, `NumPy` (Data manipulation)
  - `Matplotlib`, `Seaborn` (Visualization)
  - `Scikit-learn` (Machine learning models and metrics)
  - `NLTK` (Text preprocessing)

---

###  Exploratory Data Analysis (EDA)

####  Key Insights:
- Dataset is **balanced**, containing **20,000 positive** and **20,000 negative** reviews in the training set.
- It have balanced test set as well with **5000 positive** and **5000 negative** reviews
- However the dataset contains 272 duplicate reviews in the training set and 13 duplicate reviews in the testing set.
- **Review Length**:
  - Word count ranges: **10 – 2,500** (avg. ~300 words)
  - Character count: avg. ~1,500 characters
- No need for data balancing due to equal sentiment distribution.


####  Visualizations:
- **Histograms** for word and character counts revealed **right-skewed** distributions.
- **Word Clouds**:
  - Positive sentiments: Words like _“great”_, _“amazing”_, _“love”_ were prominent.
  - Negative sentiments: Words like _“boring”_, _“waste”_, _“bad”_ were frequent.

---

###  Text Preprocessing

####  Steps Performed:
- **Tokenization**: Splitting reviews into individual tokens.
- **Stop-word Removal**: Removed common stop-words like _“the”_, _“is”_, etc.
  
####  Stemming vs Lemmatization:
- **Stemming**: Reduced words but harmed readability (_“running” → “run”_).
- **Lemmatization**: Retained meaningful word forms and improved interpretability (_“running” → “running”_).

**Conclusion**: **Lemmatization** chosen for its semantic clarity.

---

### Vectorization

- **TF-IDF Vectorizer**:
  - Chosen over CountVectorizer due to its ability to **down-weight frequently occurring but less informative words**.
  - Enhanced focus on sentiment-bearing terms.

---


### Dimenstionality Reduction


- **Dimensionality Reduction** is performed using **Truncated SVD** (Singular Value Decomposition).
- It reduces the high-dimensional TF-IDF feature space into a more compact representation, improving model efficiency and generalization by retaining the most informative components.

---

### Modeling

#### Models Explored:
| Rank | Model                | Vectorizer | Dimensionality Reduction | Accuracy | F1 Score | Recall | Support |
|------|----------------------|------------|---------------------------|----------|----------|--------|---------|
| 1    | **Naive Bayes**      | TF-IDF     | (No reduction)         | **0.8641** | 0.86     | 0.86   | 9987    |
| 2    | **SVM**              | TF-IDF     |    Yes                     | **0.86** | 0.86     | 0.86   | 9987    |
| 3    | **Logistic Regression** | TF-IDF |      Yes                   | **0.86** | 0.86     | 0.86   | 9987    |
| 4    | Naive Bayes          | CountVec   | (No reduction)         | 0.8567   | 0.86     | 0.86   | 9987    |
| 5    | Logistic Regression  | CountVec   |        Yes                | 0.8172   | 0.82     | 0.82   | 9987    |
| 6    | Random Forest        | TF-IDF     |       Yes                 | 0.8164   | 0.82     | 0.82   | 9987    |
| 7    | Random Forest        | CountVec   |        Yes              | 0.7580   | 0.76     | 0.76   | 9987    |



#### Final Model Selection:
- **Logistic Regression** chosen due to:
  - **Highest accuracy (86%)**
  - **Balanced precision, recall, and F1-score**

---

### Overall Performance Evaluation

- **Accuracy**: 86%
- **F1-Score**: 0.86 (both classes)
- **Inference ready** and generalizes well to unseen data.

---

### Business Applications
**For Filmmakers:**
Review sentiment can be used by movie studios to know what was liked or not liked by people in a film such as the plot, acting, or music. They can make wiser choices regarding future films, sequels, or edits because of this. If they also gather more data such as age or location of viewers, then they can make films that various segments like.

**For Streaming Platforms:**
Such platforms as Netflix or Amazon Prime can utilize this analysis to suggest improved shows or determine what films to promote more. If reviews indicate that people enjoy a movie, it can be promoted; otherwise, adjustments can be made. The fact that people like something in some places can also be used to display the appropriate content, at the right time and to the correct viewers.

**For Marketing Teams:**
Marketing teams are able to review the sentiment of reviews and determine if their ads and trailers are performing effectively. If they don't do so, they can immediately make adjustments in their campaigns. This assists in spending money wiser and targeting the proper audience with the proper message.

**For Box Office Projections:**
By reading early reviews, theaters and studios can estimate how well a film will perform based on ticket sales. If a film is receiving positive reviews, they may schedule additional showtimes. In the long run, they can utilize this information to plan releases better and make more money.

**To Create New Spin-Offs or Series:**
If audiences adore a particular character or aspect of the story, filmmakers can strategize spin-offs, sequels, or even merchandise based on it. Observing which sections receive the most positive reaction keeps fans engaged and creates successful franchises.

---

##  Machine Learning Engineering (MLE) Part

### Introduction

The MLE component focuses on **operationalizing** the sentiment classification pipeline. Using the final Logistic Regression model, the objective is to implement robust **training and inference scripts**, enable **reproducibility**, and **containerize the solution** using Docker.

---

### Directory Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── processed_train.parquet
│       ├── processed_test.parquet
├── notebooks/
│   └── Final_Project.ipynb
├── outputs/
│   ├── models/
│   │   ├── logreg_model.pkl
|   |   ├── tfidf.pkl
│   ├── predictions/
│   |   ├── predictions.csv
│   |   ├── metrics.txt
│   |   ├── confusion_matrix.png
│   |   └── roc_auc_curve.png
|   └── mlruns/
├── src/
│   ├── train/
│   │   ├── training.py
│   │   └── Dockerfile
│   └── inference/
│       ├── infer.py
│       └── Dockerfile
└── requirements.txt
```



##  Workflow Overview

### Training (`train.py`)

- Loads preprocessed training data from `/data/processed/processed_train.parquet`
- Transforms text using:
  - **TF-IDF vectorization**
  - **Truncated SVD** for dimensionality reduction
- Trains a **Logistic Regression** model on the reduced features
- Evaluates the model using accuracy score
- Logs parameters, metrics, and training duration with **MLflow**
- Saves the following artifacts to `/outputs/models/`:
  - Trained model as `logreg_model.pkl`
  - TF-IDF vectorizer as `tfidf.pkl`
  - SVD transformer as `svd.pkl`

---

###  Inference (`inference.py`)

- Loads inference data from `/data/processed/processed_test.parquet`
- Loads trained artifacts from `/outputs/models/`:
  - `logreg_model.pkl`, `tfidf.pkl`, and `svd.pkl`
- Applies **TF-IDF vectorization** and **SVD** to the test data
- Performs sentiment prediction using the trained model
- If ground truth labels are available, computes:
  - Accuracy, Precision, Recall, F1-Score
  - ROC AUC score
- Logs all metrics and evaluation artifacts using **MLflow**
- Saves outputs to `/outputs/predictions/`:
  - Predictions as `predictions.csv`
  - Evaluation reports:
    - `classification_report.txt`
    - `confusion_matrix.png`
    - `roc_auc_curve.png`

---



###  Dockerization

#### **Training Docker Image**
- Built using `Dockerfile` in `src/train/` directory
- Mounts `/outputs/` to persist trained model and logs

#### **Inference Docker Image**
- Built using `Dockerfile` in `src/inference/` directory
- Outputs predictions and evaluation artifacts and saves them to  `/outputs/predictions/`

---

### How to Run

#### prerequisites

```bash
git clone https://github.com/AmbekarTejas-epam/Final_Project.git
cd Final_Project
```

Ensure **Docker** is installed: [Install Docker](https://www.docker.com/products/docker-desktop)

---
### 1. Loading Data
For loading the data from the `data\raw` directory and preprocess it run the load_data.py file from `src\` directory using the below command
```bash
python src/load_data.py
```
After running this file you can see a new directory in `data\` directory named `processed` which contains `processed_train.parquet` and `processed_test.parquet` data files. For training the model we use the processed data.


###  2. Train the Logistic Regression Model

Build the Docker Image using the below command
```bash
docker build -t sentiment-train -f src/train/Dockerfile .

```
Run the docker container using the below command for training the model
```bash
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\outputs:/app/outputs `
  -v ${PWD}\outputs\mlruns:/app/mlruns `
  sentiment-train
```

---

### 3. Run Inference
Build the docker image using the below command for inference
```bash
docker build -t sentiment-infer -f src/inference/Dockerfile .
```
Run the inference container using the below command
```bash
docker run --rm `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\outputs:/app/outputs `
  -v ${PWD}\outputs\mlruns:/app/mlruns `
  sentiment-infer

```

This saves:
- `predictions.csv`
- `metrics.txt`
- `confusion_matrix.png`
- `roc_auc_curve.png`

---
## MLflow Logging

To view logged metrics and artifacts in MLflow:

```bash 
python -m mlflow ui --backend-store-uri outputs/mlruns --port 5000
```

Open the URL provided in the terminal (e.g., `http://127.0.0.1:5000`) in a web browser. In that click on the experiment names on the sidebar and click on the latest model and go to Model metrics to view the logged metrics.

##  Model Evaluation (Final Results)

- **Classification Report**


| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.87     | 0.85   | 0.86     | 4992    |
| Positive  | 0.85      | 0.87   | 0.86     | 4995    |
| **Accuracy** |        |        | **0.86** | **9987** |
| **Macro Avg** | 0.86  | 0.86   | 0.86     | 9987    |
| **Weighted Avg** | 0.86 | 0.86 | 0.86     | 9987    |



- **Overall Accuracy**: **86%**

---

## Conclusion

The **Logistic Regression** model, supported by **TF-IDF vectorization** and **lemmatization**, provided an effective, interpretable, and deployable solution for movie review sentiment analysis. With an **89% accuracy**, it offers reliable performance in both development and production settings, enabling real-world applications in **feedback systems**, **customer support**, and **market analytics**.

---
