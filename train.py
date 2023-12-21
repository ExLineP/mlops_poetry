import os
import pickle
import string

import mlflow
import nltk
import pandas as pd
from mlflow.tracking import MlflowClient
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

nltk.download("stopwords")
# Load data
tweets_df = pd.read_csv("data/twitter.csv")
tweets_df = tweets_df.iloc[::10, :]

tweets_df.drop(columns=["id"], inplace=True)


# Text cleaning functions
def message_cleaning(message):
    message_punc_removed = [char for char in message if char not in string.punctuation]
    message_punc_removed_join = "".join(message_punc_removed)
    message_punc_removed_join_clean = [
        word
        for word in message_punc_removed_join.split()
        if word.lower() not in stopwords.words("english")
    ]
    return message_punc_removed_join_clean


# Vectorize text data
vectorizer = CountVectorizer(analyzer=message_cleaning)
tweets_countvectorizer = vectorizer.fit_transform(tweets_df["tweet"]).toarray()
X = pd.DataFrame(tweets_countvectorizer)
y = tweets_df["label"]

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.6, random_state=42
)

# MLflow setup
experiment_name = "Default"
artifact_repository = "./mlflow-run"
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
client = MlflowClient()

# Get or create experiment
try:
    experiment_id = client.create_experiment(
        experiment_name, artifact_location=artifact_repository
    )
except:
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id


def model_experimentation(classifier, model_name, run_name):
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        _ = run.info.run_uuid
        mlflow.sklearn.autolog()

        tags = {"Application": "Twitter Sentiment Analysis", "release.version": "1.0.0"}
        mlflow.set_tags(tags)

        clf = classifier
        clf.fit(X_train, y_train)

        model_filename = f"data/models/{model_name}_model.pkl"
        os.makedirs(
            os.path.dirname(model_filename), exist_ok=True
        )  # Create the directory if it doesn't exist
        with open(model_filename, "wb") as model_file:
            pickle.dump(clf, model_file)
        mlflow.log_artifact(model_filename)

        valid_prediction = clf.predict_proba(X_valid)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_valid, valid_prediction)
        roc_auc = auc(fpr, tpr)
        mlflow.log_metrics({"Validation_AUC": roc_auc})
        ConfusionMatrixDisplay.from_estimator(
            clf, X_valid, y_valid, display_labels=["Placed", "Not Placed"], cmap="magma"
        )


# Model training and evaluation
classifier = MultinomialNB()
model_name = "NB"
run_name = "NaiveBayes_model"
model_experimentation(classifier, model_name, run_name)

classifier = SVC(probability=True)
model_name = "SVC"
run_name = "SVC_model"
model_experimentation(classifier, model_name, run_name)

classifier = KNeighborsClassifier()
model_name = "KNN"
run_name = "KNN_model"
model_experimentation(classifier, model_name, run_name)

classifier = LogisticRegression()
model_name = "LogReg"
run_name = "LR_model"
model_experimentation(classifier, model_name, run_name)
