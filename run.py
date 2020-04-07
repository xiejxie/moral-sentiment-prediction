from scipy.stats import mode
from features import FeatureExtractor
from gensim.models import KeyedVectors
from senticnet.senticnet import SenticNet
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from itertools import zip_longest
from imblearn.over_sampling import RandomOverSampler
import argparse
import json
import constants
import html
import re
import numpy as np
import pandas as pd


def get_twitter_data(file_path):
    """ Parse the json tweets file located at file_path.

    return: a list of short texts, their corresponding moral labels
    """
    sentences, annotes = [], []
    with open(file_path) as src:
        data = json.load(src, encoding="utf-8")
        
        # Iterate through each individual corpus in data
        for d in data:
            for t in d["Tweets"]:
                text = t["tweet_text"]
                if text == "no tweet text available":
                    continue

                # Select mode tag as ground truth
                all_annotes = [x["annotation"] for x in t["annotations"]]
                all_annotes = ",".join(all_annotes).split(",")
                det_class = mode(all_annotes).mode[0]

                # Disregard non-moral tweets
                if det_class == "non-moral":
                    continue
                sentences.append(text)
                annotes.append(mode(all_annotes).mode[0])
    return sentences, annotes


def get_vignettes_data(file_path):
    """ Parse the json tweets file located at file_path.

    return: a list of short texts, their corresponding moral labels
    """
    df = pd.read_csv(file_path, index_col=False)
    return df["text"].values, df["category"].values


def get_data(args):
    """ Parse either tweets or vignettes data.

    return: a list of short texts, their corresponding moral labels
    """
    if args.data_type == "twitter":
        sentences, annotes = get_twitter_data(f"{args.dir}/data/{args.mtc}")
    elif args.data_type == "vignettes":
        sentences, annotes = get_vignettes_data(f"{args.dir}/data/{args.vignettes}")
    else:
        raise NotImplementedError
    return sentences, annotes


def train_test(X, X_names, X_labels, y, models, outputdir, classification_type,
    data_type, balance=False):
    """ Run k-fold cross validation for training and testing data.

    X: set of all features
    X_names: set of all feature types
    X_labels: text belonging to each x
    y: moral labels
    models: classification models
    outputdir: directory to save the output dataframes
    classification_type: {polar, categorical}
    data_type: {vignettes, twitter}
    balance: whether to correct for class imbalance
    """
    kf = KFold(n_splits=5)
    X_labels = np.array(X_labels)
    records = []
    error_inds = {"labels": [], "errors": [], "type": []}

    # Split training and testing into folds
    for i, indices in enumerate(kf.split(X[0])):
        train_i, test_i = indices
        y_train, y_test = y[train_i], y[test_i]
        
        # Iterate through each feature type set, e.g. 
        # {reasoning, both, emotion}, balance if selected
        for name, X_i in zip(X_names, X):
            X_train, X_test = X_i[train_i], X_i[test_i]
            if balance:
                r = RandomOverSampler(random_state=42)
                X_train, y_train = r.fit_resample(X_train, y_train)
            
            all_errors = np.zeros((len(X_test)))
            
            # Fit each model and predict the test accuracy
            for model in models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                all_errors += y_pred!=y_test
                records.append({
                    "model": model.__class__.__name__, "features": name,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="micro"),
                    "recall": recall_score(y_test, y_pred, average="micro"),
                    "fold": i
                })
            
            # Record where the models errored with their textual data
            error_inds["errors"].extend(all_errors/len(models))
            error_inds["labels"].extend(X_labels[test_i])
            error_inds["type"].extend([name] * len(test_i))
    
    # Save results
    records = pd.DataFrame(records)
    error_inds = pd.DataFrame(error_inds).sort_values(by="errors", ascending=False)
    error_inds[error_inds["type"] == "reasoning"].to_csv(f"{outputdir}/{data_type}_{classification_type}_errors_df.csv", index=False)
    records.to_csv(f"{outputdir}/{data_type}_{classification_type}_records_df.csv")


def preprocess(sentences, embeds, annotes, ratings_df_path):
    """ Preprocess all texts to be classified.

    sentences: a set of texts to be classified
    embeds: a dictionary of word embeddings
    annotes: moral labels
    ratings_df_path: filepath to affective ratings
    """
    def contains_emotion(list_of_words, ratings_df):
        return any(w in ratings_df.index for w in list_of_words)

    new_sentences, new_annotes = [], []
    
    # Get affective ratings
    ratings_df = pd.read_csv(ratings_df_path,
    usecols=["Word","V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"], index_col="Word")
    
    # Iterate through each piece of text and associated label
    for sentence, annote in zip(sentences, annotes):
        new_sentence = []
        sentence = html.unescape(sentence)
        for word in sentence.split():
            # Strip punctuation and non-letters from text
            new_word = re.sub('[^A-Za-z ]+', '', word).lower()
            if new_word in embeds:
                new_sentence.append(new_word)
        
        # Assert that the text contains sufficient data such that we can extract
        # emotional and reasoning based features from it
        if contains_emotion(new_sentence, ratings_df) and len(new_sentence) > 2:
            new_sentences.append(' '.join(new_sentence))
            new_annotes.append(annote)
    return new_sentences, new_annotes


def get_y(annotes, classification_type):
    """ Convert text labels to a number, indicating morally positive/negative
    or a category.

    annotes: moral labels, e.g. ["purity", "care", ...]
    classification_type: {polar, categorical}
    """
    assert classification_type in {"categorical", "polar"}
    y = np.array([constants.CATEGORIES[c] for c in annotes])
    if classification_type == "polar":
        return constants.convert_to_polar(y)
    return y


def main(args):
    embeds = KeyedVectors.load_word2vec_format(
            f"{args.dir}/data/{args.embeds}",
            binary=True, limit=args.vocab_size)

    # Do preprocessing and feature extraction
    sentences, annotes = get_data(args)
    sentences, annotes = preprocess(sentences, embeds, annotes,
        f"{args.dir}/data/ratings.csv")
    feature_extractor = FeatureExtractor(args, embeds, sentences)

    # Organize the different feature sets
    X_reasoning = feature_extractor.extract(sentences, feat_type="reasoning")
    X_emotion = feature_extractor.extract(sentences, feat_type="emotion")
    X_both = feature_extractor.extract(sentences)
    X = [X_reasoning, X_emotion, X_both]
    X_names = ["reasoning", "emotion", "both"]

    # Run training and testing for all permutations of features, models
    y = get_y(annotes, args.classification_type)
    models = [KNeighborsClassifier(), GaussianNB(), LogisticRegression(), SVC()]
    train_test(X, X_names, sentences, y, models, f"{args.dir}/results",
        args.classification_type, args.data_type, balance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", 
        help="Directory containing code and data")
    parser.add_argument("--embeds", help="Embedding package name")
    parser.add_argument("--classification_type",
        choices=["polar", "categorical"])
    parser.add_argument("--mtc", default="MFTC_V4_tweets.json",
        help="Name for moral twitter corpus json source")
    parser.add_argument("--mfd", default="mfd_v1.csv")
    parser.add_argument("--vocab_size", type=int, default=None,
        help="Vocabulary size")
    parser.add_argument("--data_type", choices=["twitter", "vignettes"])
    parser.add_argument("--vignettes", default="vignettes.csv")
    main(parser.parse_args())