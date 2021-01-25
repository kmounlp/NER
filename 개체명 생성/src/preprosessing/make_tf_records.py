import datetime
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os


BOS_ID = 1
EOS_ID = 2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(inputs_vocab, inputs_ne, inputs_pos, targets_vocab, targets_ne, targets_pos):
    feature = {
        "inputs_vocab": _int64_feature(inputs_vocab),
        "inputs_ne": _int64_feature(inputs_ne),
        "inputs_pos": _int64_feature(inputs_pos),

        "targets_vocab": _int64_feature(targets_vocab),
        "targets_ne": _int64_feature(targets_ne),
        "targets_pos": _int64_feature(targets_pos)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def make_data_set():
    pass


def create_tf_records(TF_RECORDS_PATH, aligns, min_seq_len, max_seq_len, per_file_limit=50000):
    filename = os.path.join(TF_RECORDS_PATH, str(datetime.datetime.now().timestamp()) + ".tfrecord")

    tf_writer = tf.io.TFRecordWriter(filename)
    sent_counts = 0
    total_sent_count = 0

    aligns = tqdm(aligns, desc="Creating TF Records...") # 10踰?諛섎났

    for align in aligns:

        encoded_id = align[0]
        encode_ne_id = align[1]
        encode_pos_id = align[-1]

        if max_seq_len > len(encoded_id) > min_seq_len:
            inputs_vocab = np.array([BOS_ID] + encoded_id)
            targets_vocab = np.array(encoded_id + [EOS_ID])

            inputs_ne = np.array([BOS_ID] + encode_ne_id)
            targets_ne = np.array(encode_ne_id + [EOS_ID])

            inputs_pos = np.array([BOS_ID] + encode_pos_id)
            targets_pos = np.array(encode_pos_id + [EOS_ID])

            example = serialize_example(inputs_vocab, inputs_ne, inputs_pos, targets_vocab, targets_ne, targets_pos)

            tf_writer.write(example)
            sent_counts += 1


        if sent_counts >= per_file_limit:
            tf_writer.write(example)
            total_sent_count += sent_counts
            sent_counts = 0
            tf_writer.close()
            filename = os.path.join(TF_RECORDS_PATH, str(datetime.datetime.now().timestamp()) + ".tfrecord")
            tf_writer = tf.io.TFRecordWriter(filename)

    print(sent_counts + total_sent_count)


if __name__ == '__main__':

    for _ in range(1):
        create_tf_records(10, 512) # small