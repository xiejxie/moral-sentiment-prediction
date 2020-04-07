# moral-sentiment-prediction
This repository hosts code and data that can be used to reproduce analyses and visualizations.

## Set-up
For replication, a set of word embeddings should be added under the data directory. The vectors used in the paper can be found [here](https://code.google.com/archive/p/word2vec/ "here").
All code was developed under Python 3.7.

## Running Analyses
```run.py``` facilitates all training and testing. Sets of features are extracted and classification is performed on all sets.

| Arg  | Default Value  | Description  |
| ------------ | ------------ | ------------ |
| dir  | .  |  Path to source directory containing code and data subdirs |
| embeds  | None  |  Name of embedding package |
| classification_type  | None  |  Specifies Must be one of polar or categorical |
| mtc  | MFTC_V4_tweets.json  |  Name of tweets json file |
| mfd  | mfd_v1.csv  |  Name of file with moral foundations dictionary data |
| vocab_size  |  None | Optional limit of word embedding vocab size  |
| vignettes  |  vignettes.csv | Name of file with moral vignettes data  |


For a detailed explanation of arguments, use
```
python run.py --help
```

## Visualizing Results
```visualize.py``` produces graphs and plots associated with the results from the predictive analyses. Note that this should be run after the analyses are done. In particular, ```run.py``` generates the csv files that are used as input to the visualizations.

| Arg  |  Default Value | Description  |
| ------------ | ------------ | ------------ |
| dir  | .  | Path to source directory containing code and data subdirs  |
|  data_type | None  | Specifies whether to visualize twitter data results or vignette data results. Must be one of {"twitter", "vignettes"}. |
| classification_type  | None  | Specifies whether to visualize polar or categorical results. Must be one of {"polar", "categorical"}.  |
