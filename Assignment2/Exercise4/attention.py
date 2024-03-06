from torch import nn
import torch

def embedding_of_sentence():
    sentence = 'Attention is all you need for now'
    words = sentence.split()
    sort_words = sorted(words, key=str.lower)
    T = len(words) # 7
    mapping_word_to_integer = {}
    for i, word in enumerate(sort_words):
        # dict of words mapped to int from 1 to T
        mapping_word_to_integer[word] = i + 1 
    # create tensor where each word is replaced by token
    sentence_tokenized = []
    for word in words:
        sentence_tokenized.append(mapping_word_to_integer[word])
    # create token embedding of size 16
    tensor_tokenized = torch.tensor(sentence_tokenized)
    return tensor_tokenized

embedding_of_sentence()