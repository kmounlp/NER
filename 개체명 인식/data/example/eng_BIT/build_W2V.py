"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Guillaume Genthial"

from pathlib import Path

import numpy as np
from gensim import models

if __name__ == '__main__':
    # Load vocab
    with Path('vocab.words.txt').open(encoding="utf8") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))
    embeddings = np.random.randn(size_vocab, 300) * np.sqrt(1 / 300)
    # Get relevant glove vectors
    found = 0
    print('Reading W2V file (may take a while)')
    model = models.KeyedVectors.load_word2vec_format(r'./GoogleNews-vectors-negative300.bin', binary=True)
    # model = models.KeyedVectors.load_word2vec_format(r'./word2vec_sj_kor.model', binary=False)
    # model = models.KeyedVectors.load_word2vec_format(r'./word2vec_sj_kor.model', binary=False, encoding='utf-8', unicode_errors='ignore')
##    word_vector = models.Word2Vec.load(r'./GoogleNews-vectors-negative300.bin')
    word_vector = model.wv
    OOV = []
    for word in word_to_idx.keys():

        word_idx = word_to_idx[word]
        try:
            embeddings[word_idx] = word_vector.wv.get_vector(word)
            found += 1
        except KeyError:
            OOV.append(word)
            continue
    OOV = set(OOV)
    with open("OOV.txt",'w',encoding="utf8") as wf:
        for word in OOV:
            print(word,file=wf)
    
    # with Path('glove.840B.300d.txt').open(encoding="utf8") as f:
    #     for line_idx, line in enumerate(f):
    #         if line_idx % 100000 == 0:
    #             print('- At line {}'.format(line_idx))
    #         line = line.strip().split()
    #         if len(line) != 300 + 1:
    #             continue
    #         word = line[0]
    #         embedding = line[1:]
    #         if word in word_to_idx:
    #             found += 1
    #             word_idx = word_to_idx[word]
    #             embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed('W2V.npz', embeddings=embeddings)
