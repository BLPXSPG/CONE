from pydoc import doc
from re import S 
from types import LambdaType
import torch as T
from transformers import BertTokenizer, BertModel, BertConfig
from sentence_transformers import SentenceTransformer
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
#from sentence_transformers import SentenceTransformer
import csv
from scipy.stats import gamma, lognorm, norm
import math
import random
import itertools
import json
from sklearn.decomposition import PCA
from Evaluation import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument('--n_sents', type=int, default=3)
parser.add_argument('--n_aspects', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--latent_size', type=int, default=50)
#parser.add_argument('--n_sentences', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--device', type=str, default="cuda:1")
parser.add_argument('--dataset', type=str, default="HotelRec")
parser.add_argument('--cluster', type=str, default="KMeans")
def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False
parser.add_argument('--resume_training', type=str2bool, default=False)
parser.add_argument('--to_train', type=str2bool, default=True)
parser.add_argument('--save_embedding', type=str2bool, default=True)
parser.add_argument('--to_evaluate', type=str2bool, default=False)
opt = parser.parse_args()

#load bert model and freeze it
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).to(opt.device)
sent_model = SentenceTransformer('all-MiniLM-L6-v2').to(opt.device)
evaluation = Evaluation(opt.device)

for param in bert_model.parameters():
    param.requires_grad = False
config = BertConfig()
project_dir = os.path.abspath(os.path.dirname(__file__))
if opt.dataset == "HotelRec": 
    data_path = os.path.join(project_dir, "data", "hotel")
elif opt.dataset == "VAD":
    data_path = os.path.join(project_dir, "data", "tweet")
save_path = os.path.join(data_path, 'check_point')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(os.path.join(save_path, "all_aspects_point"))
    os.makedirs(os.path.join(save_path, "all_sentiment_point"))
    os.makedirs(os.path.join(save_path, "all_docsent_point"))


class OpinionClustering():
    def __init__(self, n_aspects):
        if opt.cluster == "KMeans":
            self.aspect_model = KMeans(n_clusters=n_aspects, init='k-means++')
            self.sentiment_model = KMeans(n_clusters=2, init='k-means++')
            self.epoch = 10
        elif opt.cluster == "GMM":
            self.aspect_model = AgglomerativeClustering(n_clusters=n_aspects)
            self.sentiment_model = AgglomerativeClustering(n_clusters=2)
            self.epoch = 1
            
    def update_cluster(self, train_type, input_embeddings):
        print("start train %s cluster" % train_type)
        if train_type == "aspect":
            for i in range(self.epoch):
                self.aspect_model.fit(input_embeddings)
            if opt.cluster == "KMeans":
                cluster_centers = self.aspect_model.cluster_centers_
            elif opt.cluster == "GMM":
                cluster_centers = self.get_cluster_centers(self.aspect_model.labels_, input_embeddings)
            return self.aspect_model.labels_, cluster_centers
        elif train_type == "sentiment":
            for i in range(self.epoch):
                self.sentiment_model.fit(input_embeddings)
            if opt.cluster == "KMeans":
                cluster_centers = self.sentiment_model.cluster_centers_
            elif opt.cluster == "GMM":
                cluster_centers = self.get_cluster_centers(self.sentiment_model.labels_, input_embeddings)
            return self.sentiment_model.labels_, cluster_centers

    def cal_sentence_distance(self, senta, sentb):
        return np.linalg.norm(senta - sentb)

    def merge(self, input_embeddings, labels, cluster_centers, n_sampled=1000):
        n_points = input_embeddings.shape[0]
        n_labels = len(list(set(labels)))
        print("start calculate distribution of sentence distance")
        sentences_index = [i for i in range(n_points)]
        sampled_sentence = random.sample(sentences_index, k=n_sampled)
        distances = [self.cal_sentence_distance(input_embeddings[x[0]], input_embeddings[x[1]]) for x in itertools.combinations(sampled_sentence, r=2)]
        mean = np.mean(distances)
        var = np.var(distances)
        threshold = norm.ppf(0.05, loc=mean, scale=var**0.5)
        if threshold <= 0:
            return labels, cluster_centers
        print("sentence embedding mean: %.3f, var: %.3f, threshold: %3f" % (mean, var, norm.ppf(0.05, loc=mean, scale=var**0.5)))
        merge_remap = {}
        for i in range(n_labels):
            merge_remap[i] = i
        for x in itertools.combinations([i for i in range(n_labels)], r=2):
            embeddings_of_chosen_pair = [j for j in range(n_points) if labels[j] in x]
            sampled_sentence = random.sample(embeddings_of_chosen_pair, k=min(n_sampled, len(embeddings_of_chosen_pair)))
            distances_between = []
            for y in itertools.combinations(sampled_sentence, r=2):
                distances_between.append(self.cal_sentence_distance(input_embeddings[y[0]], input_embeddings[y[1]]))
            if np.mean(distances_between) < threshold:
                #print(np.mean(distances_between))
                map_to = min(x[0], x[1])
                while merge_remap[map_to] != map_to:
                    map_to = merge_remap[map_to]
                merge_remap[max(x[0], x[1])] = map_to
        label_left = list(set(merge_remap.values()))
        label_remap = {}
        for i in range(len(label_left)):
            label_remap[label_left[i]] = i
        labels = [label_remap[merge_remap[label]] for label in labels]
        #label_list = list(set(labels))
        print("merge result", merge_remap, label_remap, label_left, len(label_left))

        cluster_centers = [[] for i in range(len(label_left))]
        for i in range(len(labels)):
            cluster_centers[label_remap[merge_remap[labels[i]]]] += [input_embeddings[i]]
        for label_i in range(len(label_left)):
            cluster_centers[label_i] = np.mean(cluster_centers[label_i], axis=0)
        
        return labels, cluster_centers

    def get_cluster_centers(self, labels, input_embeddings):
        cluster_centers = [[] for i in range(len(labels))]
        for i in range(len(labels)):
            cluster_centers[labels[i]] += [input_embeddings[i]]
        for label_i in range(len(labels)):
            cluster_centers[label_i] = np.mean(cluster_centers[label_i], axis=0)
        return cluster_centers
        

    def sort_tag(self, labels):
        example_output = {}
        for i in range(len(labels)):
            if labels[i] in example_output:
                example_output[int(labels[i])] += [i]
            else:
                example_output[int(labels[i])] = [i]
        return example_output

    def sort_sent(self, labels, cluster_centers, input_embeddings, sort=True, n_top=50):
        n_clusters = len(list(set(labels)))
        #print("number of clusters", n_clusters)
        example_output = self.sort_tag(labels)
        labels_count = {}
        top_sentences = {}
        for aspect_index in list(set(labels)):
            if aspect_index not in example_output:
                continue
            sentence_ids = example_output[aspect_index]
            #print("aspect", aspect_index, "with #sentences", len(sentence_ids), len(sentence_ids)/len(labels))
            #calculate the % of each aspect, and % of sentiment in each aspect
            labels_count[aspect_index] = len(sentence_ids)/len(labels)
            sentence_distance = {}
            #sentiment_score = {}
            if sort:
                for j in sentence_ids:
                    sentence_distance[j] = self.cal_sentence_distance(cluster_centers[aspect_index], input_embeddings[j])
                sentence_distance = {k: v for k, v in sorted(sentence_distance.items(), key=lambda item: item[1], reverse=False)}
                sentence_ids = list(sentence_distance.keys()) 
            top_sentences[aspect_index] = sentence_ids[:n_top]
        return top_sentences, labels_count

    def pack_result(self, top_sentences, sentiment_count, aspect_pca_result, sentiment_pca_result, sentences):
        #sort output to save
        output = {}
        for sentiment_index in top_sentences:
            keypoints = [sentences[i] for i in top_sentences[sentiment_index]]
            aspect_embeddings = [aspect_pca_result[i] for i in top_sentences[sentiment_index]]
            sentiment_embeddings = [sentiment_pca_result[i] for i in top_sentences[sentiment_index]]
            output[sentiment_index] = {"percentage":sentiment_count[sentiment_index], "keypoint":keypoints, "aspect_embeddings":aspect_embeddings, "sentiment_embeddings":sentiment_embeddings}
        return output

    def top_sentences(self, aspect_labels, aspect_centers, aspect_embeddings, sent_labels, sent_centers, sent_embeddings, sentences, print_result=True):
        """
        aspectsplit: % of aspect_i
        sentimentsplit: {percentage, keypoints}
        percentage: % of sentiment
        keypoints: key points text
        embeddings: key points embedding (PCA for visualisation)
        """
        final_result = {}
        aspect_pca = PCA(n_components=3)
        aspect_pca_result = aspect_pca.fit_transform(aspect_embeddings)
        sentiment_pca = PCA(n_components=3)
        sentiment_pca_result = sentiment_pca.fit_transform(sent_embeddings)

        aspect_sentences, aspect_count = self.sort_sent(aspect_labels, aspect_centers, aspect_embeddings, n_top=50)
        for aspect_index in aspect_sentences:
            sent_labels_i = [sent_labels[i] for i in aspect_sentences[aspect_index]]
            sent_embeddings_i = [sent_embeddings[i] for i in aspect_sentences[aspect_index]]
            sentences_i = [sentences[i] for i in aspect_sentences[aspect_index]]
            sent_sentences, sentiment_count = self.sort_sent(sent_labels_i, sent_centers, sent_embeddings_i, n_top=3)
            final_result[aspect_index] = {"aspectsplit":aspect_count[aspect_index], "sentimentsplit":self.pack_result(sent_sentences, sentiment_count, aspect_pca_result, sentiment_pca_result, sentences_i)}
            if print_result:
                print(aspect_index, final_result[aspect_index])
        return final_result
    
    def aggregate_sentiment(self, sentiment_embeddings):
        n_sentences, _ = sentiment_embeddings.shape
        start = 0
        last_i = 0
        document_sentiment = []
        document_rating = []
        assert len(self.unique_documents) == len(self.rating)
        doc_count = 0
        for i, doc_id in enumerate(self.documents[:n_sentences]):
            if last_i != doc_id:
                document_sentiment.append(np.average(sentiment_embeddings[start:i], axis=0).tolist())
                document_rating.append(self.rating[doc_count])
                doc_count += 1
                last_i = doc_id
                start = i
        #print("document sentiment shape", len(document_sentiment), len(document_sentiment[0]))
        #np.save(os.path.join(save_path, "document_sentiment.npy"), np.array(document_sentiment))
        np.savetxt(os.path.join(save_path, "sentiment_point.tsv"), np.array(document_sentiment), delimiter="\t")
        np.savetxt(os.path.join(save_path, "sentiment_label.tsv"), np.array(document_rating), delimiter="\t")
    
    def save_embeddings(self):
        #save aspect embeddings
        np.savetxt(os.path.join(save_path, "aspect_point.tsv"), self.aspect_embeddings, delimiter="\t")
        #np.save(os.path.join(save_path, "aspect_embeddings.npy"), self.aspect_embeddings)
        
        #save sentiment embeddings and labels
        self.aggregate_sentiment(self.sent_embeddings)
        #np.save(os.path.join(save_path, "sent_embeddings.npy"), self.sent_embeddings)
        
    def get_disentangled_embedding(self, model, sentence_embed, batch_size = opt.batch_size):
        n_points = sentence_embed.shape[0]
        n_batches = int(n_points/batch_size)
        output_embed = []
        output_sentiment = []
        for i in range(n_batches):
            sentence_embed_i = T.from_numpy(sentence_embed[i*batch_size: (i+1)*batch_size]).to(opt.device)
            aspect_embed, sent_embed, _ = model(sentence_embed_i, 0, 0, 0, 0, train=False)
            output_embed.append(aspect_embed.cpu().detach())
            output_sentiment.append(sent_embed.cpu().detach())
        sentence_embeddings = T.cat(output_embed, 0)
        sentiment_embeddings = T.cat(output_sentiment, 0)
        return sentence_embeddings.numpy(), sentiment_embeddings.numpy()

    def train(self, model, input_embeddings, sentences, initialise=False, save_embedding=opt.save_embedding, evaluate=True):
        if initialise:
            aspect_labels, aspect_centers = self.update_cluster("aspect", input_embeddings)
            aspect_labels, aspect_centers = self.merge(input_embeddings, aspect_labels, aspect_centers)
            #aspect_sentences, aspect_count = self.sort_sent(aspect_labels, aspect_centers, input_embeddings, n_top=5)
            return aspect_labels, len(aspect_centers)
        else:
            aspect_embeddings, sent_embeddings = self.get_disentangled_embedding(model, input_embeddings)
            aspect_labels, aspect_centers = self.update_cluster("aspect", aspect_embeddings)
            aspect_labels, aspect_centers = self.merge(aspect_embeddings, aspect_labels, aspect_centers)
            sent_labels, sent_centers = self.update_cluster("sentiment", sent_embeddings)
            final_result = self.top_sentences(aspect_labels, aspect_centers, aspect_embeddings, sent_labels, sent_centers, sent_embeddings, sentences)
            if save_embedding:
                self.aspect_embeddings, self.sent_embeddings = aspect_embeddings, sent_embeddings
                np.savetxt(os.path.join(save_path, "aspects_label.tsv"), aspect_labels, delimiter="\t")
            self.aspect_centers = aspect_centers
            return aspect_labels, sent_labels, final_result, aspect_embeddings, len(aspect_centers)

    def evaluation(self, sentences, aspect_embeddings, aspect_labels, hotels, dataset="HotelRec"):
        sentiment_words_path = os.path.join(data_path, "clean", "sentiment-words.txt")
        if dataset == "HotelRec":
            #print("check hotel input", hotels[0])
            #sort by hotels
            keypoints = {}
            for i, aspect in enumerate(aspect_labels):
                hotel = hotels[i]
                if hotel not in keypoints:
                    keypoints[hotel] = {}
                    keypoints[hotel]["sentence"] = [sentences[i]]
                    keypoints[hotel]["embedding"] = [aspect_embeddings[i]]
                    keypoints[hotel]["aspect_label"] = [aspect]
                else:
                    keypoints[hotel]["sentence"] += [sentences[i]]
                    keypoints[hotel]["embedding"] += [aspect_embeddings[i]]
                    keypoints[hotel]["aspect_label"] += [aspect]
            coherences = []
            diversities = []
            distances = []
            distinct_ones = []
            distinct_twos = []
            n_kps = []
            wordoverlaps = []
            for hotel in keypoints:
                aspect_labels = keypoints[hotel]["aspect_label"]
                input_embeddings = np.vstack(keypoints[hotel]["embedding"])
                #n_top = min(6, input_embeddings.shape[0])
                aspect_sentences, _ = self.sort_sent(aspect_labels, self.aspect_centers, input_embeddings, n_top=6)
                for aspect in aspect_sentences:
                    aspect_sentences[aspect] = [sentences[i] for i in aspect_sentences[aspect]]
                coherence, uniqueness, distance, distinct_one, distinct_two, n_kp, wordoverlap = evaluation.run_eva(aspect_sentences, sentiment_words_path)
                if coherence: coherences.append(coherence)
                if uniqueness: 
                    diversities.append(uniqueness)
                    distances.append(distance)
                    wordoverlaps.append(wordoverlap)
                distinct_ones.append(distinct_one)
                distinct_twos.append(distinct_two)
                n_kps.append(n_kp)
            evaluation.print_result(coherences, diversities, distances, distinct_ones, distinct_twos, n_kps, wordoverlaps)
        elif dataset == "VAD":
            evaluation.output_eva_twitter(sentences, aspect_labels, sentiment_words_path)


class ConeDataset(Dataset):
    def __init__(self, split = 0.9, feature="bert_tokenizer", filter_neutral=False):
        self.split = split
        self.feature = feature
        self.silhouette_score = []
        self.n_aspects = opt.n_aspects
        self.silhouette = 0
        self.dataset_chosen = opt.dataset
        print("dataset chosen", self.dataset_chosen)

        #load data
        self.sentences, _len = self.load_data("reviews.txt")
        self.sentences_aug, sent_len_aug = self.load_data("reviews_aug_tran.txt")
        self.sentiment_tags, self.unique_sentiments = self.load_tag("sentiments.npy", type="sentiment")
        assert _len == sent_len_aug
        print("number of sentences", _len)
        self.sentences, self._len, self.sentences_aug, self.sentiment_tags = self.filter_sentences(self.sentences, _len, self.sentences_aug, self.sentiment_tags, filter_neutral)
        if self.dataset_chosen == "HotelRec":
            self.hotels, self.unique_hotels = self.load_tag("hotels.npy", type="document")
            self.rating, self.unique_rating = self.load_tag("document_rating.npy", type="sentiment")
            self.documents, self.unique_documents = self.load_tag("documents.npy", type="document")
            self.documents = self.documents[:self._len]
        elif self.dataset_chosen == "VAD":
            self.hotels = T.zeros(self._len)
            self.documents = T.zeros(self._len)
        self.sentence_embed = self.embed_text(self.sentences)
        self.sentence_embed_aug = self.embed_text(self.sentences_aug)
        self.update_aspect(None, self.sentence_embed, initialise=True)
        #print(len(self.aspect_tags), len(self.sentiment_tags), self._len)
        assert len(self.aspect_tags) == self._len
        assert len(self.sentiment_tags) == self._len

    def load_data(self, filename):
        data_dir = os.path.join(data_path, "clean", filename)
        with open(data_dir, 'r') as f:
            data = f.readlines()
            f.close()
        n_sent = len(data)
        return data, n_sent
        
    def load_tag(self, filename, type="sentiment"):
        data_dir = os.path.join(data_path, "clean", filename)
        data = np.array(np.load(data_dir))#[:self.sent_len]
        if type == "sentiment":
            data += 1
        unique_tags = list(set(data))
        print("number of", filename, len(unique_tags))
        return T.LongTensor(data), unique_tags

    def filter_sentences(self, sentences, _len, sentences_aug, sentiment_tags, filter_neutral):
        if filter_neutral:
            sentences_filtered = []
            sentences_aug_filtered = []
            sentiment_tags_filtered = []
            for i in range(_len):
                if sentiment_tags[i] == 1:
                    continue
                elif sentiment_tags[i] == 2:
                    sentences_filtered.append(sentences[i])
                    sentences_aug_filtered.append(sentences_aug[i])
                    sentiment_tags_filtered.append(1)
                elif sentiment_tags[i] == 0:
                    sentences_filtered.append(sentences[i])
                    sentences_aug_filtered.append(sentences_aug[i])
                    sentiment_tags_filtered.append(0)
            assert len(sentences_filtered) == len(sentences_aug_filtered)
            print(sentences_filtered[:3])
            print(sentences_aug_filtered[:3])
            print(sentiment_tags_filtered[:3])
            print(T.LongTensor(sentiment_tags_filtered)[:3])
            _len = int(len(sentences_filtered)/opt.batch_size)*opt.batch_size
            return sentences_filtered[:_len], _len, sentences_aug_filtered[:_len], T.LongTensor(sentiment_tags_filtered)[:_len]
        else:
            _len = int(len(sentences)/opt.batch_size)*opt.batch_size
            return sentences[:_len], _len, sentences_aug[:_len], T.LongTensor(sentiment_tags)[:_len]

    def cal_sentence_distance(self, senta, sentb):
        #print("sent norm term", (np.linalg.norm(senta) * np.linalg.norm(sentb)))
        #return np.inner(senta, sentb) / (np.linalg.norm(senta) * np.linalg.norm(sentb))
        return np.linalg.norm(senta - sentb)

    def embed_text(self, sentences, batch_size = opt.batch_size):
        sentence_embeddings = sent_model.encode(sentences)
        return sentence_embeddings

    def update_aspect(self, model, input_embedding, initialise=False):
        cluster_model = OpinionClustering(self.n_aspects)
        if initialise:
            aspect_tags, self.n_aspects = cluster_model.train(model, input_embedding, self.sentences, initialise=initialise)
            self.aspect_tags = T.LongTensor(aspect_tags)
        else:
            aspect_tags, sentiment_tags, final_result, aspect_embedding, self.n_aspects = cluster_model.train(model, self.sentence_embed, self.sentences, initialise=initialise)
            self.aspect_tags = T.LongTensor(aspect_tags)
            self.sentiment_tags = T.LongTensor(sentiment_tags)
            if opt.to_evaluate:
                print("start evaluation:")
                
                cluster_model.evaluation(self.sentences, aspect_embedding, aspect_tags, self.hotels.numpy(), dataset = self.dataset_chosen)
            score = silhouette_score(aspect_embedding, aspect_tags)
            print("silhouette score", self.silhouette, score)
            if self.silhouette >= score:
                return True
            else:
                self.silhouette = score
                return False

    def __getitem__(self, item):
        return self.sentence_embed[item], self.sentence_embed_aug[item], self.aspect_tags[item], self.sentiment_tags[item], self.sentences[item], self.documents[item]

    def __len__(self):
        return self._len


class MLP(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim_a = 128, hidden_dim_b = 64):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_dim, hidden_dim_a),
      nn.ReLU(),
      nn.Linear(hidden_dim_a, hidden_dim_b),
      nn.ReLU(),
      nn.Linear(hidden_dim_b, output_dim)
    )

  def forward(self, x):
    return self.layers(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, doc_info=True, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", T.tensor(temperature))
        self.register_buffer("negatives_mask", (~T.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.device = device
        self.doc_info = doc_info
            
    def forward(self, emb_i, emb_j, tags, num_classes, document_ids):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = T.cat([z_i, z_j], dim=0)
        
        tags = F.one_hot(tags, num_classes=num_classes)
        tagcombine = T.cat([tags, tags], dim=0)
        
        if self.doc_info:
            n_ids = max(document_ids) + 1
            doc_ids = F.one_hot(document_ids, num_classes=n_ids)
            doc_ids = T.cat([doc_ids, doc_ids], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        def l_ij(i, j):
            numerator = T.exp(similarity_matrix[i,j]/self.temperature)
            tag_mask = T.sum(T.square(tagcombine - tagcombine[i])/2, dim=1)
            mask = tag_mask
            if self.doc_info:
                doc_mask = T.sum(T.square(doc_ids - doc_ids[i])/2, dim=1)
                mask = tag_mask*doc_mask
            denominator = T.sum(
                mask * T.exp(similarity_matrix[i, :] / self.temperature)
            ) + 0.1
            
            loss_ij = -T.log(numerator / denominator)
            return loss_ij.squeeze(0)

        loss = 0.0
        for n in range(self.batch_size):
            loss += l_ij(n, n + self.batch_size) + l_ij(n + self.batch_size, n)
        return loss / (2 * self.batch_size)

class DisentangleModel(nn.Module):
    def __init__(self, config, opt):
        super().__init__()
        self.device = opt.device
        self.n_sentiment = opt.n_sents
        self.n_aspect = opt.n_aspects
        self.sentiment_mlp = MLP(opt.hidden_size, opt.hidden_size)
        self.aspect_mlp = MLP(opt.hidden_size, opt.hidden_size)
        self.aspect_center = nn.Linear(opt.hidden_size, opt.latent_size)
        self.softmax = nn.Softmax()
        if opt.dataset == "HotelRec": 
            doc_info = True
        elif opt.dataset == "VAD":
            doc_info = False
        self.contra_loss = ContrastiveLoss(opt.batch_size, self.device, doc_info=doc_info)

    def forward(self, sentence_embed, sentence_embed_aug, aspect_tags, sentiment_tags, document_ids, train = True):
        """ 
        x is the input embedding with shape [bs*nsents, n_words, n_feature]
        sent_mask is the input embedding with shape [bs*nsents]
        doc_rating is the ground truth doc rating with shape [n_doc]
        """
        #mask = sent_mask.type(T.FloatTensor)
        total_loss = 0

        #if self.embed_type == "mlp":
        aspect_embed = self.aspect_mlp(sentence_embed)
        sent_embed = self.sentiment_mlp(sentence_embed)
        if train:
            aspect_embed_aug = self.aspect_mlp(sentence_embed_aug)
            sent_embed_aug = self.sentiment_mlp(sentence_embed_aug)
        
            contrastive_loss_aspect = self.contra_loss(aspect_embed, aspect_embed_aug, aspect_tags, self.n_aspect, document_ids)
            contrastive_loss_sentiment = self.contra_loss(sent_embed, sent_embed_aug, sentiment_tags, self.n_sentiment, document_ids)
            contrastive_loss =  contrastive_loss_sentiment + contrastive_loss_aspect
            total_loss += contrastive_loss

        return aspect_embed, sent_embed, total_loss
    
if __name__ == "__main__":
    cone_dataset = ConeDataset()
    cone_dataloader = DataLoader(cone_dataset, batch_size=opt.batch_size, shuffle=True)
    model = DisentangleModel(config, opt).to(opt.device)
    trainer_vae = T.optim.Adam(model.parameters(), lr=opt.lr)

    aspects_all = []
    sents_all = []
    aspects_label = []
    sents_label = []
    sentences_text_all = []
    n_steps = int(cone_dataset._len/opt.batch_size)-1
    for epoch in range(opt.n_epoch):
        if (epoch > 1) and epoch%5==0:
            stop = cone_dataset.update_aspect(model, None)
            if stop: 
                print("early stop")
                break
        for step, (sentence_embed, sentence_embed_aug, aspect_tags, sentiment_tags, sentences_text, document_ids) in enumerate(cone_dataloader):
            sentence_embed, sentence_embed_aug = sentence_embed.to(opt.device), sentence_embed_aug.to(opt.device)
            aspect_tags, sentiment_tags, document_ids = aspect_tags.to(opt.device), sentiment_tags.to(opt.device), document_ids.to(opt.device)
            aspect_embed, sent_embed, loss = model(sentence_embed, sentence_embed_aug, aspect_tags, sentiment_tags, document_ids)
            trainer_vae.zero_grad()
            loss.backward()
            trainer_vae.step()
            if (step % 1000 == 0) or (step == n_steps): 
                print("epoch: %d, step: %d, loss: %.3f" %(epoch, step, loss))
            if epoch == opt.n_epoch - 1:
                aspects_all.append(aspect_embed)
                sents_all.append(sent_embed)
                aspects_label.append(aspect_tags)
                sents_label.append(sentiment_tags)
                sentences_text_all += sentences_text
                if (step % 500 == 0) or (step == n_steps): 
                    all_aspects_point = T.cat(aspects_all, 0).cpu().detach().numpy()
                    all_sentiment_point = T.cat(sents_all, 0).cpu().detach().numpy()
                    all_aspects_label = T.cat(aspects_label, 0).cpu().detach().numpy()
                    all_sents_label = T.cat(sents_label, 0).cpu().detach().numpy()

                    np.save(os.path.join(save_path, "all_aspects_point.npy"), all_aspects_point)
                    np.save(os.path.join(save_path, "all_sentiment_point.npy"), all_sentiment_point)
                    np.save(os.path.join(save_path, "aspects.npy"), all_aspects_label)
                    np.save(os.path.join(save_path, "sentiments.npy"), all_sents_label)
                    stripped_review = [sent.replace("\n", ' ').replace("\r", ' ') for sent in sentences_text_all]
                    np.savetxt(os.path.join(save_path, "reviews.txt"),
                            stripped_review,
                            fmt="%s")

                    aspects_all = []
                    sents_all = []
                    aspects_label = []
                    sents_label = []
                    sentences_text_all = [] 
                    
    cone_dataset.update_aspect(model, None)