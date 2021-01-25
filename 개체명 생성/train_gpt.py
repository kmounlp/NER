import glob
from src.gpt_model import *
from utils.data_pipeline import input_fn
from configuration import Config
import os
import json
import pickle


def train(config):
    par_map = {"num_layers": config.num_layers, "d_model": config.d_model,
               "n_model": config.n_model, "p_model": config.p_model,
               "num_heads": config.num_heads, "dff": config.dff,
               "max_seq_len": config.max_seq_len, "vocab_size": config.VOCAB_SIZE, "ne_size": config.NE_SIZE, "pos_size": config.POS_SIZE}

    exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    with open(config.MODEL_DIR + '/model_par.json', 'w') as f:
        json.dump(par_map, f)

    tf_records = glob.glob((config.TF_RECORDS_PATH+"/*.tfrecord"))
    train_percent = int(len(tf_records) * 0.8)
    train_tf_records = tf_records[:train_percent]
    test_tf_records = tf_records[train_percent:]

    train_dataset = input_fn(train_tf_records, batch_size=config.batch_size, shuffle=True, epoch=config.epoch)
    test_dataset = input_fn(test_tf_records, batch_size=config.batch_size, shuffle=True, epoch=config.epoch)

    model = Gpt2(config)
    model.create_optimizer()
    model.create_checkpoint_manager(config.MODEL_DIR)
    model.create_summary_writer(config.LOG_DIR)
    model.fit([train_dataset, test_dataset])
    model.summary()

    print("Training Done................")

if __name__ == "__main__":
    from src.preprosessing.prepare_utils import load_dict
    config = Config()
    config.POS_SIZE = len(load_dict(config.POS_DICT_PATH))

    if config.use_BIT:
        config.NE_SIZE = len(load_dict(config.NE_POS_DICT_PATH))
    else:
        config.NE_SIZE = len(load_dict(config.NE_DICT_PATH))

    # print(config.POS_SIZE)
    # print(config.NE_SIZE)
    train(config)
