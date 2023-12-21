import os
import pickle
import string

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


# Load trained models
def load_model(model_name):
    model_filename = f"data/models/{model_name}_model.pkl"
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)
    return model


# Text cleaning function
def message_cleaning(message):
    message_punc_removed = [char for char in message if char not in string.punctuation]
    message_punc_removed_join = "".join(message_punc_removed)
    message_punc_removed_join_clean = [
        word
        for word in message_punc_removed_join.split()
        if word.lower() not in stopwords.words("english")
    ]
    return message_punc_removed_join_clean


# Load the vectorizer and fit_transform with training data
vectorizer = CountVectorizer(analyzer=message_cleaning)
tweets_df = pd.read_csv("data/twitter.csv")
tweets_df = tweets_df.iloc[::10, :]
X_train = vectorizer.fit_transform(tweets_df["tweet"]).toarray()

# Example text data for inference
text_data = [
    "This is a positive tweet.",
    "Negative sentiment is not good.",
    "Neutral statement here.",
]

# Transform text data
text_countvectorizer = vectorizer.transform(text_data).toarray()


# Inference function
def make_predictions(model, data):
    predictions = model.predict_proba(data)[:, 1]
    return predictions


# Load models
nb_model = load_model("NB")
svc_model = load_model("SVC")
knn_model = load_model("KNN")
lr_model = load_model("LogReg")

# Make predictions
nb_predictions = make_predictions(nb_model, text_countvectorizer)
svc_predictions = make_predictions(svc_model, text_countvectorizer)
knn_predictions = make_predictions(knn_model, text_countvectorizer)
lr_predictions = make_predictions(lr_model, text_countvectorizer)

# Create a DataFrame to store the results
results_df = pd.DataFrame(
    {
        "Text": text_data,
        "NaiveBayes_Predictions": nb_predictions,
        "SVC_Predictions": svc_predictions,
        "KNN_Predictions": knn_predictions,
        "LogReg_Predictions": lr_predictions,
    }
)

directory = "data/results"
if not os.path.exists(directory):
    os.makedirs(directory)

results_df.to_csv(f"{directory}/inference_results.csv", index=False)

print("Inference results saved to 'inference_results.csv'.")
