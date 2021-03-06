from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import json
import constants
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(y_true, y_preds, model):
    """ Plot confusion matrix between truth and predictions for the best
    performing model
    """
    conf_matrix = confusion_matrix(y_true, y_preds)
    ax = sns.heatmap(conf_matrix, xticklabels=constants.CATEGORY_LABELS,
        yticklabels=constants.CATEGORY_LABELS)
    ax.set_title(model)
    ax.set_xticklabels(constants.CATEGORY_LABELS, rotation=30)
    plt.show()


def plot_visualize_features(args):
    """ Wrapper function for visualize_features
    """
    npzfile = np.load(f"{args.dir}/results/{args.data_type}_categorical.npz")
    feature_type = "both"
    X_train, y_train = npzfile[feature_type], npzfile["y_train"]
    visualize_features(X_train, y_train, feature_type)


def visualize_features(X, y, feature_type):
    """ Visualize (using TSNE in 2D space) the features for both reasoning
    and emotion combined.
    """
    # Reduce features to 2D space
    X_red = TSNE(n_components=2).fit_transform(X)
    # Convert labels from their numeric equivalents, [0,4,...] to their text
    # labels, ["care", "loyalty",...]
    labels = constants.convert_to_labels(y)
    
    # Make scatterplot
    ax = sns.scatterplot(x=X_red[:,0], y=X_red[:,1], hue=labels, hue_order=constants.CATEGORY_LABELS,
        palette=sns.color_palette("Paired", 10))
    ax.set_title(f"Twitter features ({feature_type})")
    plt.tight_layout()
    plt.show()


def select_best_model(records_df):
    """ Select model with highest performing accuracy

    records_df: dataframe recording accuracy by model/model features
    """
    i = records_df["accuracy"].idxmax()
    return records_df.at[i, "model"], records_df.at[i, "features"]


def plot_categorical(args):
    """ Run the confusion plot visualization
    """
    # Read in csv data from `run.py`
    categorical_preds_df = pd.read_csv(f"{args.dir}/results/twitter_categorical_preds_df.csv")
    categorical_records_df = pd.read_csv(f"{args.dir}/results/twitter_categorical_records_df.csv")
    
    # Select top performing classification model + feature set
    model, features = select_best_model(categorical_records_df)
    
    # Make confusion matrix
    y_preds = categorical_preds_df[f"{model}_{features}"].values
    y_true = categorical_preds_df["truth"].values
    plot_confusion_matrix(y_true, y_preds, f"{model} ({features})")


def plot_records(args):
    """ Plot accuracy graphs. In particular, produce bar plots that display
    accuracy of each model while accounting for feature type.
    """
    # Read csvs data produced in `run.py`
    twitter_records_df = pd.read_csv(f"{args.dir}/results/twitter_polar_records_df.csv")
    twitter_c_records_df = pd.read_csv(f"{args.dir}/results/twitter_categorical_records_df.csv")
    vignettes_records_df = pd.read_csv(f"{args.dir}/results/vignettes_categorical_records_df.csv")
    df = pd.concat([twitter_c_records_df, vignettes_records_df], ignore_index=True)

    # Assign column data to differentiate the data types
    vignettes_records_df["data"] = "vignettes"
    twitter_c_records_df["data"] = "twitter"

    # Shorten model names for ease of reading
    short_names = {
        "KNeighborsClassifier": "kNN",
        "GaussianNB": "NB",
        "LogisticRegression": "LR",
        "SVC": "SVC"
    }
    df["model"] = df["model"].map(short_names)
    twitter_records_df["model"] = twitter_records_df["model"].map(short_names)

    # Make plots
    if args.classifcation_type == "categorical":
        sns.catplot(x="model", y="accuracy",
                    hue="features", col="data",
                    data=df, kind="bar")
        plt.show()
    else:
        sns.catplot(x="model", y="accuracy",
                    hue="features",
                    data=twitter_records_df, kind="bar")
        plt.title("Twitter Polar Classification")
        plt.show()


def count_twitter_text(args):
    """ Count total number of tweets available from twitter data.
    """
    no_text_available = 0
    totals = 0
    with open(f"{args.dir}/data/MFTC_V4_tweets.json") as src:
        data = json.load(src, encoding="utf-8")
        for d in data:
            for t in d["Tweets"]:
                text = t["tweet_text"]
                if text == "no tweet text available":
                    no_text_available += 1
                totals += 1
    print(no_text_available)
    print(totals)
    print(totals-no_text_available)



def main(args):
    # Set higher font scale for ease of reading in paper format
    sns.set(font_scale=1.7)
    plot_records(args)
    if args.classifcation_type == "categorical":
        plot_categorical(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory containing code and data")
    parser.add_argument("--data_type", choices = ["twitter", "vignettes"])
    parser.add_argument("--classifcation_type", choices=["polar", "categorical"])
    main(parser.parse_args())