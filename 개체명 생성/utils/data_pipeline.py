import collections
import glob
import tensorflow as tf
#
#
# def load_vocab(vocab_path):
#     vocab = collections.OrderedDict()
#     index = 0
#     for line in open(vocab_path, 'r').read().splitlines():
#         vocab[line.split()[0]] = index
#         index += 1
#     inv_vocab = {v: k for k, v in vocab.items()}
#     return vocab, inv_vocab
#
#
# def convert_by_vocab(vocab, items):
#     output = []
#     for item in items:
#         output.append(vocab[item])
#     return output
#
#
# def convert_tokens_to_ids(vocab, tokens):
#     return convert_by_vocab(vocab, tokens)
#
#
# def convert_ids_to_tokens(inv_vocab, ids):
#     return convert_by_vocab(inv_vocab, ids)


def parse_example(serialized_example):
    data_fields = {
        "inputs_vocab": tf.io.VarLenFeature(tf.int64),
        "inputs_pos": tf.io.VarLenFeature(tf.int64),
        "inputs_ne": tf.io.VarLenFeature(tf.int64),

        "targets_vocab": tf.io.VarLenFeature(tf.int64),
        "targets_pos": tf.io.VarLenFeature(tf.int64),
        "targets_ne": tf.io.VarLenFeature(tf.int64)
    }

    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs_vocab = tf.sparse.to_dense(parsed["inputs_vocab"])
    inputs_pos = tf.sparse.to_dense(parsed["inputs_pos"])
    inputs_ne = tf.sparse.to_dense(parsed["inputs_ne"])

    targets_vocab = tf.sparse.to_dense(parsed["targets_vocab"])
    targets_pos = tf.sparse.to_dense(parsed["targets_pos"])
    targets_ne = tf.sparse.to_dense(parsed["targets_ne"])

    inputs_vocab = tf.cast(inputs_vocab, tf.int32)
    inputs_pos = tf.cast(inputs_pos, tf.int32)
    inputs_ne = tf.cast(inputs_ne, tf.int32)

    targets_vocab = tf.cast(targets_vocab, tf.int32)
    targets_pos = tf.cast(targets_pos, tf.int32)
    targets_ne = tf.cast(targets_ne, tf.int32)

    return inputs_vocab, inputs_pos, inputs_ne, targets_vocab, targets_pos, targets_ne


def input_fn(tf_records, batch_size=32, padded_shapes=([-1], [-1], [-1], [-1], [-1], [-1]), epoch=10, buffer_size=10000, shuffle=True):
    if type(tf_records) is str:
        tf_records = [tf_records]
    dataset = tf.data.TFRecordDataset(tf_records, buffer_size=10000)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.map(parse_example)
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

