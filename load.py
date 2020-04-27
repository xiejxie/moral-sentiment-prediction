import json
from scipy.stats import mode
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
    
    # TODO: Implement classification_type
    return sentences, annotes


def get_vignettes_data(file_path, vignette_type, classification_type):
    """ Parse the json tweets file located at file_path.

    return: a list of short texts, their corresponding moral labels
    """
    df = pd.read_csv(file_path, index_col=False)
    df = df[df["dataset"] == vignette_type]

    if classification_type == "polar":
        return df["text"].values, df["polarity"].values
    return df["text"].values, df["category"].values