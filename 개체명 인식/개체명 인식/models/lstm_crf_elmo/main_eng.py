"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tf_metrics import precision, recall, f1
from bilm.data import Batcher
from bilm.model import BidirectionalLanguageModel
from bilm.elmo import weight_layers

# from allennlp.modules.elmo import Elmo, batch_to_ids
# from bilm.model import BidirectionalLanguageModel
# from bilm.elmo import weight_layers
DATADIR = '../../data/example/eng_tag'
# options_file = r"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# weight_file = r"options.json"
# Logging
Path('results_eng_tag').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results_eng_tag/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    # batcher = Batcher(r"C:\Users\NLP-Ho\Downloads\bilm-tf-master\bin\vocab\vocab", 40)
    # temp = [w for w in line_words.strip().split()]
    # char_ids = batcher.batch_sentences([temp])[0]
    # chars = char_ids.tolist()

    assert len(words) == len(tags), "Words and tags lengths don't match"
    return (words, len(words)), tags


def generator_fn(words, tags):
    with Path(words).open('r',encoding="utf8") as f_words, Path(tags).open('r',encoding="utf8") as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']), features['elmo_input'])

    # Read vocabs and inputs
    dropout = params['dropout']
    (words, nwords) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open(encoding="utf8") as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    # 한글
    # options_file = r"C:\Users\NLP-Ho\Downloads\bilm-tf-master\output_path\to\checkpoint\options.json"
    # weight_file = r"C:\Users\NLP-Ho\Downloads\bilm-tf-master\output_path\to\weights.hdf5"
    # bilm = BidirectionalLanguageModel(options_file=options_file, weight_file=weight_file, use_character_inputs=True)
    # ops = bilm(elmo_inputs)
    # weight_op = weight_layers("nerelmo", ops)['weighted_op']

    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(
        inputs={
            "tokens": words,
            "sequence_len": nwords
        },
        signature="tokens",
        as_dict=True)["elmo"]
    # Word Embeddings
    # from allennlp.modules.elmo import Elmo, batch_to_ids
    #
    # options_file = r"C:\Users\NLP-Ho\Downloads\bilm-tf-master\output_path\to\checkpoint\options.json"
    # weight_file = r"C:\Users\NLP-Ho\Downloads\bilm-tf-master\output_path\to\weights.hdf5"
    #
    # elmo = Elmo(options_file, weight_file, 2, dropout=0)
    #
    # # use batch_to_ids to convert sentences to character ids
    # character_ids = batch_to_ids(words)
    # # print(character_ids[0].shape)
    # # print(len(character_ids))
    #
    # embeddings = elmo(character_ids)
    # print(embeddings['elmo_representations'])
    # word_ids = vocab_words.lookup(words)
    # BiLM = BidirectionalLanguageModel(options_file, weight_file)
    # ops = BiLM(word_ids)
    # weight_op = weight_layers("name", ops)['weighted_op']

    # glove = np.load(params['W2V'])['embeddings']  # np.array
    # variable = np.vstack([glove, [[0.]*params['dim']]])
    # variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    # embeddings = tf.nn.embedding_lookup(variable, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    # t = tf.transpose(weight_op, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'lstm_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'W2V': str(Path(DATADIR, 'W2V.npz'))
    }
    with Path('results_eng_tag/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))

    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))

    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=False)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=1500, session_config=gpuconfig)
    estimator = tf.estimator.Estimator(model_fn, 'results_eng_tag/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=1500)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=300)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('results_eng_tag/score').mkdir(parents=True, exist_ok=True)
        with Path('results_eng_tag/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')
    for name in ['train', 'testa', 'testb']:
        write_predictions(name)



