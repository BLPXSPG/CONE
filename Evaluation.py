from sentence_transformers import SentenceTransformer
import numpy as np
import string
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class Evaluation():
    def __init__(self, device):
        self.device = device
        self.model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    
    def cal_sentence_distance(self, senta, sentb):
        return np.linalg.norm(senta - sentb)
    
    def topic_coherence(self, sentence_embeddings):
        L, _ = sentence_embeddings.shape
        similarity = 0
        for i in range(L):
            #similarity += np.average(self.cal_sentence_distance([sentence_embeddings[i]], np.delete(sentence_embeddings, i, 0)))
            similarity += np.average(cosine_similarity([sentence_embeddings[i]], np.delete(sentence_embeddings, i, 0)))
        return similarity/L
    
    def topic_uniqueness(self, sentence_embeddings):
        L, _ = sentence_embeddings.shape
        uniqueness = 0
        similarity = 0
        distance = 0
        for i in range(L):
            uniqueness_i = np.reciprocal(cosine_similarity([sentence_embeddings[i]], np.delete(sentence_embeddings, i, 0)))
            uniqueness += np.average(uniqueness_i)
            distance_i = np.linalg.norm(np.delete(sentence_embeddings, i, 0) - [sentence_embeddings[i]], axis=0)
            distance += np.average(distance_i)
            #similarity += np.average(cosine_similarity([sentence_embeddings[i]], np.delete(sentence_embeddings, i, 0)))
        return uniqueness/L, distance/L

    def topic_diversity(self, sents):
        sents = [sentence.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) for sentence in sents]
        sents = [sentence.translate(str.maketrans('', '', '1234567890')) for sentence in sents]
        words = []
        unigrams = []
        bigrams = []
        for sentence in sents:
            tokens = [word for word in word_tokenize(sentence) if word not in stopwords.words('english')]
            words += tokens
            unigrams += list(ngrams(tokens, 1))
            bigrams += list(ngrams(tokens, 2))
        distinct_one = len(list(set(unigrams)))/len(words)
        distinct_two = len(list(set(bigrams)))/len(words)
        return distinct_one, distinct_two
    
    def word_overlap(self, top_sents, sentiment_words_path, vocab_limit = 2000):
        with open(sentiment_words_path, "r") as f:
            sentiment_words = f.readlines()
            sentiment_words = [word.strip() for word in sentiment_words]
            #print("sentiment words:", sentiment_words[:5])
            f.close()
        words_all = []
        n_aspects = len(list(top_sents.keys()))
        vocab_limit = int(vocab_limit/n_aspects)
        for aspect in top_sents:
            sents = top_sents[aspect]
            word_list = []
            for sent in sents:
                word_list += word_tokenize(sent)
            word_list = dict(Counter(word_list))
            word_list = [k for k, v in sorted(word_list.items(), key=lambda item: item[1])]
            filtered_words = [word for word in word_list if word not in stopwords.words('english')]
            #filtered_words = [word for word in word_list if word not in sentiment_words]
            words_all += filtered_words[:vocab_limit]
        """for i, words in enumerate(aspect_words):
            word_list = aspect_words[:]
            del word_list[i]
            flat_word_list = [x for xs in word_list for x in xs]"""

        word_count = dict(Counter(words_all))
        L = len(word_count.keys())
        cnt = 0
        for word in word_count:
            cnt += 1/word_count[word]
        return cnt/L
    
    def run_eva(self, top_sents, sentiment_words_path):
        coherence_cnt = []
        embedding_center = []
        diversity_one = []
        diversity_two = []
        for aspect in top_sents:
            sents = top_sents[aspect]
            distinct_one, distinct_two = self.topic_diversity(sents)
            diversity_one.append(distinct_one)
            diversity_two.append(distinct_two)
            sentence_embeddings = self.model.encode(sents)
            if len(sents) > 1:
                coherence_cnt.append(self.topic_coherence(sentence_embeddings))
            embedding_center.append(np.average(sentence_embeddings, axis=0))
        #print(embedding_center.shape)
        if len(embedding_center) > 1:
            uniqueness, distance = self.topic_uniqueness(np.stack(embedding_center, axis=0))
            wordoverlap = self.word_overlap(top_sents, sentiment_words_path)
        else: 
            uniqueness = False
            distance = False
            wordoverlap = False
        #print(uniqueness.shape)
        if len(coherence_cnt) > 0:
            coherence = sum(coherence_cnt)/len(coherence_cnt)
        else:
            coherence = False
        distinct_one = sum(diversity_one)/len(diversity_one)
        distinct_two = sum(diversity_two)/len(diversity_two)
        return coherence, uniqueness, distance, distinct_one, distinct_two, len(list(top_sents.keys())), wordoverlap
    
    def print_result(self, coherence, uniqueness, distance, distinct_one, distinct_two, n_kp, wordoverlap, resize=True):
        if resize:
            print("coherence", sum(coherence)/len(coherence))
            print("uniqueness", sum(uniqueness)/len(uniqueness))
            print("distance", sum(distance)/len(distance))
            print("distinct_ones", sum(distinct_one)/len(distinct_one))
            print("distinct_twos", sum(distinct_two)/len(distinct_two))
            print("ave #key points", sum(n_kp)/len(n_kp))
            print("word overlap", sum(wordoverlap)/len(wordoverlap))
        else:
            print("coherence", coherence)
            print("uniqueness", uniqueness)
            print("distance", distance)
            print("distinct_ones", distinct_one)
            print("distinct_twos", distinct_two)
            print("ave #key points", n_kp)
            print("word overlap", wordoverlap)
        
    def output_eva_twitter(self, sentences, aspects, sentiment_words_path, load=False): 
        if load:
            aspects = list(np.load(aspects))
            with open(sentences, 'r') as f:
                sentences = f.readlines()
                f.close()
        keypoints = {}
        print("length check",len(aspects), len(sentences))
        for i, aspect in enumerate(aspects):
            sentence = sentences[i]
            if aspect not in keypoints:
                keypoints[aspect] = [sentence]
            else:
                keypoints[aspect] += [sentence]
        coherence, uniqueness, distance, distinct_one, distinct_two, n_kp, wordoverlap = self.run_eva(keypoints, sentiment_words_path)
        self.print_result(coherence, uniqueness, distance, distinct_one, distinct_two, n_kp, wordoverlap, resize=False)
        return coherence, uniqueness, distance, distinct_one, distinct_two, n_kp, wordoverlap