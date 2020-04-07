from senticnet.senticnet import SenticNet
from gensim import corpora, models
import constants
import numpy as np
import pandas as pd

NUM_FOUNDATIONS = 10
VALID_TYPES = {"reasoning", "emotion", "both"}
sn = SenticNet()

class FeatureExtractor:
    def __init__(self, args, embeds, sentences):
        mfd = pd.read_csv(f"{args.dir}/data/{args.mfd}")
        mfd["category"] -= 1
        mfd.set_index("category")
        self.mfd = mfd
        self.ratings_df = pd.read_csv(f"{args.dir}/data/ratings.csv",
            usecols=["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"],
            index_col="Word")
        self.embeds = embeds
        self.centroids = self.__make_centroids()
        self.tfidf, self.dictionary = self.__make_tfidf(sentences)
        E = self.get_embeds(sentences)

        # Extraction
        self.features = {
            "reasoning": self.distance_feats(E),
            "emotion": self.emotion_feats(sentences)
        }
        self.features["both"] = np.concatenate(
            (self.features["reasoning"], self.features["emotion"]), axis=1)


    def __make_tfidf(self, sentences):
        texts = [document.split() for document in sentences]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(corpus)
        return tfidf[corpus], dictionary


    def extract(self, sentences, feat_type="both"):
        assert feat_type in VALID_TYPES, "Feature type not supported"
        return self.features[feat_type]


    def get_embeds(self, sentences):
        def get_sentence_embed(sentence):
            return [self.embeds[word] for word in sentence], np.arange(len(sentence))
        X = self.__vectorize(get_sentence_embed, sentences, self.embeds.vector_size)
        return X


    def distance_feats(self, E):
        assert E.ndim == 2
        X_expd = np.repeat(np.expand_dims(E, 1), NUM_FOUNDATIONS, axis=1)
        centroids_expd = np.repeat(np.expand_dims(self.centroids, 0),
            E.shape[0], axis=0)
        assert X_expd.shape == centroids_expd.shape
        distances = np.sqrt(np.sum(np.square(X_expd-centroids_expd), axis=2))
        # num = np.sum(X_expd + centroids_expd, axis=2)
        # denom = np.sqrt(np.sum(np.square(X_expd), axis=2))*np.sqrt(np.sum(np.square(centroids_expd), axis=2))
        # distances = num/denom
        return np.nan_to_num(distances)


    def emotion_feats(self, sentences):
        def get_embed(sentence):
            # labels = ["aptitude", "attention", "pleasantness", "sensitivity"]
            # e_indices = [i for i in range(len(sentence)) if sentence[i] in sn.data]
            # emotions = [[float(sn.sentics(sentence[i])[l]) for l in labels] for i in e_indices]
            labels = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
            e_indices = [i for i in range(len(sentence)) if sentence[i] in self.ratings_df.index]
            emotions = [[self.ratings_df.at[sentence[i], l] for l in labels] for i in e_indices]
            assert len(emotions) > 0, "Emotion features cannot be extracted"
            return emotions, e_indices
        return self.__vectorize(get_embed, sentences, 3)


    def __make_centroids(self):
        centroids = np.zeros((NUM_FOUNDATIONS, self.embeds.vector_size))
        counts = np.zeros((NUM_FOUNDATIONS))
        for row in self.mfd.itertuples():
            word = row.word
            if word in self.embeds:
                centroids[row.category] += self.embeds[word]
                counts[row.category] += 1
        counts = np.repeat(np.expand_dims(counts, 1), centroids.shape[1],
            axis=1)
        return centroids/counts


    def __vectorize(self, sent_transformer, sentences, V):
        N = len(sentences)
        X = np.empty((N, V))
        for i in range(len(sentences)):
            sentence = sentences[i].split()
            all_embeds, indices = sent_transformer(sentence)
            doc_tfidf = dict(self.tfidf[i])
            weights = [doc_tfidf[self.dictionary.token2id[sentence[j]]] for j in indices]
            weights = np.exp(weights)/sum(np.exp(weights))
            weights = np.repeat(np.expand_dims(weights, 1), V, 1)
            assert len(all_embeds) > 0
            X[i] = np.sum(weights*all_embeds, 0)
        return X