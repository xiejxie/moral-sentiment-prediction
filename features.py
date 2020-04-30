from senticnet.senticnet import SenticNet
from gensim.models import KeyedVectors
from gensim import corpora, models
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from collections import namedtuple
import stanza
import constants
import numpy as np
import pandas as pd

NUM_FOUNDATIONS = 10
VALID_TYPES = {"reasoning", "emotion", "both"}
sn = SenticNet()

class FeatureExtractor(ABC):
    
    @abstractmethod
    def extract(self, sentences):
        pass


class BasicFeatureExtractor(FeatureExtractor):

    def __init__(self, default_size):
        self.default_size = default_size

    @abstractmethod
    def get_features_from_word(self, word):
        pass

    @abstractmethod
    def is_word_in_vocab(self, word):
        pass

    def extract(self, sentences):
        X = []
        for sentence in sentences:
            features = [
                np.array(self.get_features_from_word(w))
                for w in sentence
                if self.is_word_in_vocab(w)
            ]
            if len(features) == 0:
                features = [np.random.randn(self.default_size)]
            X.append(np.mean(features, axis=0))
        
        return np.array(X)



class BertFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self.model = SentenceTransformer("bert-base-nli-mean-tokens")
        super().__init__()
    
    def extract(self, sentences):
        return self.model.encode(sentences)


class GloveFeatureExtractor(BasicFeatureExtractor):

    def __init__(self, embed_path):
        self.embeds = KeyedVectors.load_word2vec_format(embed_path, binary=True)
        super().__init__(self.embeds.vector_size)


    def get_features_from_word(self, word):
        return self.embeds[word]


    def is_word_in_vocab(self, word):
        return word in self.embeds
        

class MainVerbFeatureExtractor(FeatureExtractor):

    def __init__(self, embed_path):
        super().__init__()
        self.embeds = KeyedVectors.load_word2vec_format(embed_path, binary=True)
        self.nlp = stanza.Pipeline(lang='en',
            processors='tokenize,mwt,pos,lemma,depparse')

    def extract(self, sentences):
        text = " ".join(sentences)
        doc = self.nlp(text)
        X = []

        for sentence in doc.sentences:
            root = next((word.text for word in sentence.words if word.deprel == "root"), None)
            if root in self.embeds:
                X.append(self.embeds[root])
            else:
                X.append(np.random.randn(self.embeds.vector_size))
        
        return np.array(X)


class EmotionFeatureExtractor(BasicFeatureExtractor):

    def __init__(self, ratings_path):
        labels = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
        self.labels = labels
        self.ratings_df = pd.read_csv(ratings_path,
            usecols=["Word"] + labels,
            index_col="Word")
        super().__init__(len(labels))
    

    def get_features_from_word(self, word):
        return [
            self.ratings_df.at[word, l]
            for l in self.labels
        ]


    def is_word_in_vocab(self, word):
        return word in self.ratings_df


class ReasoningFeatureExtractor(GloveFeatureExtractor):

    def __init__(self, embed_path, mfd_path):
        super().__init__(embed_path)
        mfd = self.__get_mfd(mfd_path)
        self.centroids = self.__make_centroids(self.embeds, mfd)
        self.default_size = self.centroids.shape[0]


    def extract(self, sentences):
        X = super().extract(sentences)
        X_expd = np.repeat(np.expand_dims(X, 1), NUM_FOUNDATIONS, axis=1)
        centroids_expd = np.repeat(np.expand_dims(self.centroids, 0),
            X.shape[0], axis=0)
        assert X_expd.shape == centroids_expd.shape
        distances = np.sqrt(np.sum(np.square(X_expd-centroids_expd), axis=2))
        return np.nan_to_num(distances)


    def __make_centroids(self, embeds, mfd):
        centroids = np.zeros((NUM_FOUNDATIONS, embeds.vector_size))
        counts = np.zeros((NUM_FOUNDATIONS))
        for row in mfd.itertuples():
            word = row.word
            if word in embeds:
                centroids[row.category] += embeds[word]
                counts[row.category] += 1
        counts = np.repeat(np.expand_dims(counts, 1), centroids.shape[1],
            axis=1)
        return centroids/counts
    

    def __get_mfd(self, mfd_path):
        mfd = pd.read_csv(mfd_path)
        mfd["category"] -= 1
        mfd.set_index("category")
        return mfd

# class FeatureExtractor:
#     def __init__(self, args, embeds, sentences):
#         mfd = pd.read_csv(f"{args.dir}/data/{args.mfd}")
#         mfd["category"] -= 1
#         mfd.set_index("category")
#         self.mfd = mfd
#         self.ratings_df = pd.read_csv(f"{args.dir}/data/ratings.csv",
#             usecols=["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"],
#             index_col="Word")
#         self.embeds = embeds
#         self.centroids = self.__make_centroids()
#         self.tfidf, self.dictionary = self.__make_tfidf(sentences)
#         E = self.get_embeds(sentences)

#         # Extraction
#         self.features = {
#             "reasoning": self.__distance_feats(E),
#             "emotion": self.__emotion_feats(sentences)
#         }


#     def get_embeds(self, sentences):
#         def get_sentence_embed(sentence):
#             return [self.embeds[word] for word in sentence], np.arange(len(sentence))
#         X = self.__vectorize(get_sentence_embed, sentences, self.embeds.vector_size)
#         return X

    
#     def get_features(self, feat_type):
#         if feat_type == "reasoning":
#             return self.__distance_feats(E)
#         else:
#             return self.__


#     def __distance_feats(self, E):
#         assert E.ndim == 2
#         X_expd = np.repeat(np.expand_dims(E, 1), NUM_FOUNDATIONS, axis=1)
#         centroids_expd = np.repeat(np.expand_dims(self.centroids, 0),
#             E.shape[0], axis=0)
#         assert X_expd.shape == centroids_expd.shape
#         distances = np.sqrt(np.sum(np.square(X_expd-centroids_expd), axis=2))
#         return np.nan_to_num(distances)


#     def __emotion_feats(self, sentences):
#         def get_embed(sentence):
#             labels = ["V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]
#             e_indices = [i for i in range(len(sentence)) if sentence[i] in self.ratings_df.index]
#             emotions = [[self.ratings_df.at[sentence[i], l] for l in labels] for i in e_indices]
#             assert len(emotions) > 0, "Emotion features cannot be extracted"
#             return emotions, e_indices
#         return self.__vectorize(get_embed, sentences, 3)


#     def __make_centroids(self):
#         centroids = np.zeros((NUM_FOUNDATIONS, self.embeds.vector_size))
#         counts = np.zeros((NUM_FOUNDATIONS))
#         for row in self.mfd.itertuples():
#             word = row.word
#             if word in self.embeds:
#                 centroids[row.category] += self.embeds[word]
#                 counts[row.category] += 1
#         counts = np.repeat(np.expand_dims(counts, 1), centroids.shape[1],
#             axis=1)
#         return centroids/counts


#     def __vectorize(self, sent_transformer, sentences, V):
#         N = len(sentences)
#         X = np.empty((N, V))
#         for i in range(len(sentences)):
#             sentence = sentences[i].split()
#             all_embeds, indices = sent_transformer(sentence)
#             doc_tfidf = dict(self.tfidf[i])
#             weights = [doc_tfidf[self.dictionary.token2id[sentence[j]]] for j in indices]
#             weights = np.exp(weights)/sum(np.exp(weights))
#             weights = np.repeat(np.expand_dims(weights, 1), V, 1)
#             assert len(all_embeds) > 0
#             X[i] = np.sum(weights*all_embeds, 0)
#         return X


#     def __make_tfidf(self, sentences):
#         texts = [document.split() for document in sentences]
#         dictionary = corpora.Dictionary(texts)
#         corpus = [dictionary.doc2bow(text) for text in texts]
#         tfidf = models.TfidfModel(corpus)
#         return tfidf[corpus], dictionary