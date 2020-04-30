from features import BertFeatureExtractor, GloveFeatureExtractor, \
MainVerbFeatureExtractor, EmotionFeatureExtractor, ReasoningFeatureExtractor
from senticnet.senticnet import SenticNet
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from itertools import zip_longest
from imblearn.over_sampling import RandomOverSampler
from load import get_twitter_data, get_vignettes_data
import argparse
import constants
import html
import re
import os
import numpy as np
import pandas as pd


def train_test(X, feature_type, X_labels, y, models, outputdir, classification_type,
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
    kf = KFold(n_splits=5, shuffle=True)
    X_labels = np.array(X_labels)
    records = []
    X = np.array(X)
    y = np.array(y)

    # Split training and testing into folds
    for i, indices in enumerate(kf.split(X)):
        train_i, test_i = indices
        y_train, y_test = y[train_i], y[test_i]
        
        X_train, X_test = X[train_i], X[test_i]
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
                "model": model.__class__.__name__,
                "features": feature_type,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="micro"),
                "recall": recall_score(y_test, y_pred, average="micro"),
                "fold": i,
                "dataset": data_type,
            })
    
    # Save results
    save_records(records,
        f"{outputdir}/{classification_type}_records_df.csv")


def save_records(records, filepath):
    updated_records = pd.DataFrame(records)

    if os.path.exists(filepath):
        orig_records = pd.read_csv(filepath)
        updated_records = pd.concat([orig_records, updated_records], ignore_index=True)\
            .drop_duplicates(["model", "features", "fold", "dataset"],
                keep="last")

    updated_records.to_csv(filepath, index=False)


def preprocess(sentences):
    """ Preprocess all texts to be classified.

    sentences: a set of texts to be classified
    annotes: moral labels
    """
    new_sentences = []
    
    # Iterate through each piece of text and associated label
    for sentence in sentences:
        new_sentence = []
        sentence = html.unescape(sentence)
        for word in sentence.split():
            # Strip punctuation and non-letters from text
            new_word = re.sub('[^A-Za-z ]+', '', word).lower()
            new_sentence.append(new_word)
        # Assert that the text contains sufficient data such that we can extract
        # emotional and reasoning based features from it
        new_sentences.append(' '.join(new_sentence) + '.')
    return new_sentences


def init_feature_extractor(args):
    data_dir = f"{args.dir}/data"
    if args.feature_type == "bert":
        return BertFeatureExtractor()
    elif args.feature_type == "glove":
        return GloveFeatureExtractor(f"{data_dir}/{args.embeds}")
    elif args.feature_type == "main_verb":
        return MainVerbFeatureExtractor(f"{data_dir}/{args.embeds}")
    elif args.feature_type == "emotion":
        return EmotionFeatureExtractor(f"{data_dir}/ratings.csv")
    elif args.feature_type == "reasoning":
        return ReasoningFeatureExtractor(f"{data_dir}/{args.embeds}",
            f"{data_dir}/{args.mfd}")
    else:
        raise NotImplementedError


def get_data(data_type, args):
    """ Parse either tweets or vignettes data.

    return: a list of short texts, their corresponding moral labels
    """
    if data_type == "twitter":
        sentences, annotes = get_twitter_data(f"{args.dir}/data/{args.mtc}")
    elif data_type.startswith("vignettes"):
        vignette_type = data_type.split('_')[1]
        sentences, annotes = get_vignettes_data(
            f"{args.dir}/data/{args.vignettes}", 
            vignette_type, args.classification_type)
    else:
        raise NotImplementedError
    return sentences, annotes


def get_y(annotes):
    """ Convert text labels to a number, indicating morally positive/negative
    or a category.

    annotes: moral labels, e.g. ["purity", "care", ...]
    classification_type: {polar, categorical}
    """
    labels = sorted(list(set(annotes)))
    labels_dict = { label:i for i,label in enumerate(labels) }
    y = [labels_dict[annote] for annote in annotes]
    return y


def main(args):

    feature_extractor = init_feature_extractor(args)
    for data_type in args.data_types:
        # Do preprocessing and feature extraction
        sentences, annotes = get_data(data_type, args)
        sentences = preprocess(sentences)

        # Organize the different feature sets
        X = feature_extractor.extract(sentences)

        # Run training and testing for all permutations of features, models
        y = get_y(annotes)
        models = [KNeighborsClassifier(), GaussianNB(), LogisticRegression(), SVC()]
        train_test(X, args.feature_type, sentences, y, models,
            f"{args.dir}/results", args.classification_type, data_type,
            balance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", 
        help="Directory containing code and data")
    parser.add_argument("--embeds", help="Embedding package name")
    parser.add_argument("--classification_type",
        choices=["polar", "categorical"])
    parser.add_argument("--feature_type", choices=["bert", "glove",
        "main_verb", "emotion", "reasoning"])
    parser.add_argument("--mtc", default="MFTC_V4_tweets.json",
        help="Name for moral twitter corpus json source")
    parser.add_argument("--mfd", default="mfd_v1.csv")
    parser.add_argument("--vocab_size", type=int, default=None,
        help="Vocabulary size")
    parser.add_argument("--vignettes", default="vignettes.csv")
    parser.add_argument("--data_types", choices=["twitter", "vignettes_mccurrie",
        "vignettes_chadwick", "vignettes_clifford"], nargs="*")
    main(parser.parse_args())