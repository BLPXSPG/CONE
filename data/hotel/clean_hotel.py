import os
import numpy as np
import json
from operator import add
import math
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from random import shuffle

def del_if_exist(saving_path):
    if os.path.isfile(saving_path):
        os.remove(saving_path)


def rescore(score):
    if (score == 1) or (score == 2):
        return -1
    elif score == 3:
        return 0
    elif (score == 4) or (score == 5):
        return 1


def get_sentiment(sentence, score, weight = 0.5, threshold = 0.1):
    sid = SentimentIntensityAnalyzer()
    sent_score = sid.polarity_scores(sentence)['compound']
    score = (score-3)/2
    final_score = weight*sent_score + (1-weight)*score
    if final_score > threshold: 
        return 1
    elif final_score < -threshold:
        return -1
    else:
        return 0 


def text_clean(review, min_len = 10, max_length = 32):
    #review = review.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    review = review.translate(str.maketrans('', '', '1234567890'))
    #review = review.lower()
    tokenized_reviews = review.split(' ')
    #tokenized_reviews = tokenizer.tokenize(review)
    if (len(tokenized_reviews) <= min_len) or (len(tokenized_reviews) > max_length):
        return " ", False
    else:
        return review, True


def check_aspect(property_dict):
    key_list = ["sleep quality", "value", "rooms", "service", "cleanliness", "location"]
    for aspect in key_list:
        if aspect in property_dict:
            property_dict[aspect] = int(property_dict[aspect])
        else:
            return property_dict, False
    return property_dict, True


def preprocess(data_dir, save_dir, review_limit=1000):
    stats_path = os.path.join(save_dir, "stats.txt")
    del_if_exist(stats_path)
    with open(data_dir, "r") as f:
        data = f.readlines()
        shuffle(data)
        f.close()
    sentences = []
    sentiments = []
    timestamp = []
    raw_sent = []
    documents = []
    raw_doc = []
    doc_rating = []
    hotel_map = {}
    hotel = []
    #aspect rating
    sleep_quality = []
    value = []
    rooms = []
    service = []
    cleanliness = []
    location = []
    count = 0
    count_sent = [0]*5
    hotel_index = 0
    for i in range(len(data)):
        review = json.loads(data[i])
        rating = int(review["rating"])
        if count_sent[int(rating-1)] >= review_limit:
            continue
        aspect, check = check_aspect(review["property_dict"])
        #split to sentences
        sent_list = sent_tokenize(review["text"])
        if check and (len(sent_list)>5):
            flag = False
            for sent in sent_list:
                sent_clean, ifkeep = text_clean(sent)
                if ifkeep:
                    if review["hotel_url"] not in hotel_map:
                        hotel_map[review["hotel_url"]] = hotel_index
                        hotel_index += 1
                    hotel.append(hotel_map[review["hotel_url"]])
                    flag = True
                    sentences.append(sent_clean)
                    sentiments.append(get_sentiment(sent, rating))
                    timestamp.append(review["date"])
                    raw_sent.append(sent)
                    documents.append(count)
            if flag:
                count_sent[int(rating-1)] += 1
                raw_doc.append(review["text"])
                doc_rating.append(rescore(rating))
                sleep_quality.append(aspect["sleep quality"])
                value.append(aspect["value"])
                rooms.append(aspect["rooms"])
                service.append(aspect["service"])
                cleanliness.append(aspect["cleanliness"])
                location.append(aspect["location"])
                count += 1
        if i % 5000 == 0:
            assert len(set(documents)) == len(doc_rating)
            assert len(sentences) == len(sentiments)
            print("preprocess reviews", i, count, count_sent, len(set(documents)), len(doc_rating), len(sentences), len(sentiments))
        if count >= review_limit*5:
            save_data(save_dir, sentences, sentiments, raw_sent, documents, raw_doc, doc_rating, sleep_quality, value, rooms, service, cleanliness, location, hotel, hotel_map)
            return 0


def save_data(save_dir, reviews, sentiments, raw_sent, documents, raw_doc, doc_rating, sleep_quality, value, rooms, service, cleanliness, location, hotel, hotel_map):
    # Save data.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # `reviews.npy` is a [num_sentences, max_sentence_len] matrix where each
    # entry is a list of tokenizer id and padding.
    stripped_review = [sent.replace("\n", ' ').replace("\r", ' ') for sent in reviews]
    np.savetxt(os.path.join(save_dir, "reviews.txt"),
               stripped_review,
               fmt="%s")

    # `sentiments.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the sentiment label of the corresponding review.
    sentiments = np.array(sentiments)
    np.save(os.path.join(save_dir, "sentiments.npy"), sentiments)

    # `timestamps.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the post time of the corresponding review.
    #np.save(os.path.join(save_dir, "timestamps.npy"), timestamps)

    # `raw_sentences.txt` contains all the sentences we ended up using.
    stripped_sent = [sent.replace("\n", ' ').replace("\r", ' ') for sent in raw_sent]
    np.savetxt(os.path.join(save_dir, "raw_sentences.txt"),
               stripped_sent,
               fmt="%s")

    # `documents.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the documents of the corresponding review.
    np.save(os.path.join(save_dir, "documents.npy"), documents)

    # `documents_map.txt` contains all the documents sent_to_doc map to.
    stripped_sent = [sent.replace("\n", ' ').replace("\r", ' ') for sent in raw_doc]
    np.savetxt(os.path.join(save_dir, "documents_map.txt"),
               stripped_sent,
               fmt="%s")

    # `document_rating.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the sentiment label of the corresponding review.
    np.save(os.path.join(save_dir, "document_rating.npy"), doc_rating)


    # `sleep_quality.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the sleep_quality rating of the corresponding review.
    sleep_quality = np.array(sleep_quality)
    np.save(os.path.join(save_dir, "sleep_quality.npy"), sleep_quality)

    # `value.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the value rating of the corresponding review.
    value = np.array(value)
    np.save(os.path.join(save_dir, "value.npy"), value)

    # `rooms.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the rooms rating of the corresponding review.
    rooms = np.array(rooms)
    np.save(os.path.join(save_dir, "rooms.npy"), rooms)

    # `service.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the service rating of the corresponding review.
    service = np.array(service)
    np.save(os.path.join(save_dir, "service.npy"), service)

    # `cleanliness.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the cleanliness rating of the corresponding review.
    cleanliness = np.array(cleanliness)
    np.save(os.path.join(save_dir, "cleanliness.npy"), cleanliness)

    # `location.npy` is a [num_sentences] vector where each entry is an
    # integer indicating the location rating of the corresponding review.
    location = np.array(location)
    np.save(os.path.join(save_dir, "location.npy"), location)

    # `hotels.npy` contains all the hotels sent_to_doc map to.
    np.save(os.path.join(save_dir, "hotels.npy"), hotel)

    hotel_map = dict((v,k) for k,v in hotel_map.items())
    with open(os.path.join(save_dir, "hotel_map.json"), 'w') as f:
        json.dump(hotel_map, f)
        f.close()
    print("saved")


if __name__ == "__main__":
    #parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    project_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, "HotelRec", "HotelRec.txt")
    save_dir = os.path.join(project_dir, "clean")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("start to preprocess data")

    preprocess(data_dir, save_dir)

