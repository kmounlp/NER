"""Script to build words, chars and tags vocab"""

__author__ = "Guillaume Genthial"

import unicodedata
from collections import Counter
from pathlib import Path

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return '{}.words.txt'.format(name)
    def poss(name):
        return '{}.pos.txt'.format(name)
    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'testa', 'testb']:
        with Path(words(n)).open(encoding="utf8") as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path('vocab.words.txt').open('w',encoding="utf8") as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path('vocab.chars.txt').open('w',encoding="utf8") as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Jasos
    # Get all jaso
    print('Build vocab jasos')
    vocab_jasos = set()

    for c in vocab_chars:
        for jaso in unicodedata.normalize("NFD", c):
            vocab_jasos.update(jaso)

    with Path('vocab.jasos.txt').open('w',encoding="utf8") as f:
        for c in sorted(list(vocab_jasos)):
            f.write('{}\n'.format(c))

    print('- done. Found {} chars'.format(len(vocab_jasos)))
    # 4. Tags
    # Get all tags from the training set

    def tags(name):
        return '{}.tags.txt'.format(name)

    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    vocab_pos =set()
    with Path(tags('train')).open(encoding="utf8") as f:
        for line in f:
            vocab_tags.update(line.strip().split())
    for n in ['train', 'testa', 'testb']:
        with Path(poss(n)).open(encoding="utf8") as f:
            for line in f:
                vocab_pos.update(line.strip().split())
    with Path('vocab.tags.txt').open('w',encoding="utf8") as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    with Path('vocab.pos.txt').open('w',encoding="utf8") as f:
        for t in sorted(list(vocab_pos)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))
