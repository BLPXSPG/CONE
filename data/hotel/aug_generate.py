import os
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import nltk

#project_dir = os.path.abspath(os.path.dirname(__file__))
#model_path = os.path.join(project_dir, "clean", "mlm_finetuned")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('./clean/mlm_finetuned')
model.eval()

def predict_masked_sent(text, top_k=3, n_mask = 5):
    # Tokenize input
    #text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    nouns_index = [i for i, (word, pos) in enumerate(nltk.pos_tag(tokenized_text)) if(pos[:2] == 'NN')]
    #sent_len = len(tokenized_text) - 2
    masked_index = []
    #for j in range(min(n_mask, int(sent_len/3))):
    for random_index in nouns_index:
        #random_index = randrange(sent_len)
        tokenized_text[random_index] = "[MASK]"
        masked_index.append(random_index+1)
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]

    #masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    for i, mask_i in enumerate(masked_index):
        probs = torch.nn.functional.softmax(predictions[0, mask_i], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        tokenized_text[mask_i] = tokenizer.convert_ids_to_tokens([top_k_indices[-1]])[0]

    return tokenizer.convert_tokens_to_string(tokenized_text[1:-1])


def generate(data, save_dir):
    aug_sents = []
    for sent_index, sent in enumerate(data):
        aug_sent = predict_masked_sent(sent, top_k=1)
        aug_sents.append(aug_sent)
        if sent_index % 1000 == 0:
            print("input sentence:", sent.strip())
            print("aug sentence:", aug_sent)
            print()
        
    save_data(save_dir, aug_sents)
    print("finish augmentation")


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
    save_dir = os.path.join(project_dir, "clean", "reviews_aug_gen.txt")

    print("start to aug data")
    with open(data_dir, 'r') as f:
        data = f.readlines()
        f.close()

    generate(data, save_dir)




