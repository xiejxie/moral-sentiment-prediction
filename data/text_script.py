import json
import sys
from collections import OrderedDict
from twython import Twython, TwythonError, TwythonRateLimitError
from tqdm import tqdm
import time
import pandas as pd

#Fill your Twitter API key and access token below
#Information can be found on the Twitter developer website:
#https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html
consumer_key='ssWxErPmmILVQkRzZgTmfVyvU'
consumer_secret='lzlDXhJq3aozv7PSHAkBB8n7PrpaFkTE6Gq6AHM6FTnKmo6Dl5'
access_token_key='1241428130914537477-ZWPFMi574uzAAAr02naUNatacISNcK'
access_token_secret='HdFMoQO27f0UfhGJ6HfWi17SPxoT1HHYaO5KhNwPSYQbm'

def parse(input, output, corpus_name="all"):
    try:
        temp = open(input)
    except:
        print("Please enter a valid input file")

    data = json.load(temp)
    counter = 0

    for d in data:
        if((corpus_name != "all") and (corpus_name != d["Corpus"])):
            continue
        #Skip over corpus without tweet ID's
        if(d["Corpus"] == "Davidson"):
            if(corpus_name == "Davidson"):
                print("Tweet text is not availble for the Davidson corpus. "
                      "You can use the tweet_ids and retrieve the text from "
                      "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv")
            continue
        for t in tqdm(d["Tweets"]):
            id = t["tweet_id"]
            #retrieves tweet text if available
            while True:
                try:
                    text, date = call_twitter_api(id)
                    t["tweet_text"] = text
                    t["date"] = date
                    break
                except TwythonError as e:
                    t["tweet_text"] = 'no tweet text available'
                    t["date"] = 'no date available'
                    if isinstance(e, TwythonRateLimitError):
                        print("Rate error sleeping")
                        time.sleep(900)
                    else:
                        break
            #reodering the json elements
            temp = t["annotations"]
            t.pop("annotations")
            t["annotations"] = temp

    check = False
    if(corpus_name != "all"):
        for d in data:
            if(d["Corpus"] == corpus_name):
                data = d
                check = True
                break
        if(check == False):
            print("Please enter a valid corpus name")

    with open(output, 'w') as outfile:
        json.dump(data, outfile, indent = 4)


def call_twitter_api(id):
    twitter = Twython(consumer_key, consumer_secret, access_token_key, access_token_secret)
    tweet = twitter.show_status(id=id)
    return tweet['text'], tweet['created_at']


def parse_davidson(input, output, davidson_file):
    with open(input) as src, open(davidson_file) as davidson:
        data = json.load(src)
        davidson_table = pd.read_csv(davidson, index_col=0)
        for d in data:
            if(d["Corpus"] != "Davidson"):
                continue
            for t in tqdm(d["Tweets"]):
                id = int(t["tweet_id"])
                #retrieves tweet text if available
                t["tweet_text"] = davidson_table.loc[id,"tweet"]
                temp = t["annotations"]
                t.pop("annotations")
                t["annotations"] = temp
    with open(output, 'w') as outfile:
        json.dump(data, outfile, indent = 4)


if __name__ == "__main__":
    # parse(sys.argv[1], sys.argv[2])
    parse_davidson(sys.argv[1], sys.argv[2], sys.argv[3])
