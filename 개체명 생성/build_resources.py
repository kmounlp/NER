from configuration import Config
from src.preprosessing.prepare_train import train, make_dict
from src.preprosessing.prepare_utils import load_dict
from src.preprosessing.align_units import align_bp_ne_pos
from utils.data_pipeline import input_fn
import glob
import tensorflow as tf
from src.preprosessing.make_tf_records import create_tf_records

if __name__ == '__main__':
    config = Config()

    train(config=config)

    if config.is_pretraining_mode:
        make_dict(config)
    #
    aligns = align_bp_ne_pos(config)
    create_tf_records(config.TF_RECORDS_PATH, aligns, min_seq_len=10, max_seq_len=512, per_file_limit=1024)
    #
    # pos_dict = load_dict(config.REVERSE_POS_DICT_PATH)
    # ne_dict = load_dict(config.REVERSE_NE_POS_DICT_PATH)
    #
    #
    # input_fn(glob.glob((config.TF_RECORDS_PATH+"/*.tfrecord")))
    # dataset = input_fn(glob.glob((config.TF_RECORDS_PATH+"/*.tfrecord")))
    #
    # for inp_vocab, inp_pos, inp_ne, tar_vocab, tar_pos, tar_ne in dataset:
    #     for nn, pp in zip(inp_ne, inp_pos):
    #         for n, p in zip(nn, pp):
    #             # tf.print(n, p)
    #             if int(n) < 0 or int(n) >= len(ne_dict) :
    #                 print("??????????????")
    #                 break
    #             if ne_dict[int(n)] == '<pad>': break
    #             print(ne_dict[int(n)], pos_dict[int(p)], sep='\t')
    #         input("="*10)