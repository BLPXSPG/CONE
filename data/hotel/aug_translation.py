import os
import numpy as np
from googletrans import Translator


def generate(data_dir, save_dir, n_mask = 5, n_gen = 10):
    with open(data_dir, 'r') as f:
        data = f.readlines()
        f.close()
    translator = Translator()

    aug_sents = []
    for sent in data:
        trans_sent = translator.translate(sent, src='en', dest='zh-cn')
        aug_sent = translator.translate(trans_sent.text, src='zh-cn', dest='en')
        aug_sent = aug_sent.text
        aug_sents.append(aug_sent)

    save_data(save_dir, aug_sents)
    print("finish augmentation")

    for i in range(10):
        print("input sentence:", data[i].strip())
        print("aug sentence:", aug_sents[i])
        print()
        

def save_data(save_dir, reviews_aug):
    # `reviews_aug.npy` is a [num_sentences, max_sentence_len] matrix where each
    # entry is a list of tokenizer id and padding.
    stripped_aug_review = [sent.replace("\n", ' ').replace("\r", ' ') for sent in reviews_aug]
    np.savetxt(os.path.join(save_dir),
               stripped_aug_review,
               fmt="%s")


if __name__ == "__main__":
    #parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    project_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, "clean", "reviews.txt")
    save_dir = os.path.join(project_dir, "clean", "reviews_aug_tran.txt")

    print("start to aug data")

    generate(data_dir, save_dir)




