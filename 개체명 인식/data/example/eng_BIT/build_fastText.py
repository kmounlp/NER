"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Guillaume Genthial"

from pathlib import Path

import numpy as np
import fasttext

if __name__ == '__main__':
    # Load vocab
    with Path('vocab.words.txt').open(encoding="utf8") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading fasttext file (may take a while)')
    model = fasttext.load_model("crawl-300d-2M-subword.bin")
    for word in word_to_idx.keys():

        word_idx = word_to_idx[word]
        try:
            embeddings[word_idx] = model.get_word_vector(word)
            found += 1
        except KeyError:
            continue

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
    np.savez_compressed('fasttext.npz', embeddings=embeddings)
