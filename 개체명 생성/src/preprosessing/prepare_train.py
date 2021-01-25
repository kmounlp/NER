# -*-encoding: utf-8-*-
import sys
import os
# if os.path.dirname(os.path.abspath(os.path.dirname(__file__))) not in sys.path:
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
import io
import csv
import sentencepiece as spm
from .load_data import readData
from .prepare_utils import save_dict
from ftfy import fix_text
from collections import Counter

PAD_ID = 0 # <pad>
BOS_ID = 1 # <bos>
EOS_ID = 2 # <eos>
UNK_ID = 3 # </unk>

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

def make_dict(config):
    special_tokens = config.special_tokens

    NE_LIST = sorted(list(set(open(config.PROCESS_NE_PATH, mode='r', encoding='utf8').read().replace("\n", " ").split())))
    POS_LIST = sorted(list(set(open(config.PROCESS_POS_PATH, mode='r', encoding='utf8').read().replace("\n", " ").split())))
    NE_POS_LIST = sorted(NE_LIST + POS_LIST)

    ne_dict = {}
    reverse_ne_dict = {}
    pos_dict = {}
    reverse_pos_dict = {}
    ne_pos_dict = {}
    reverse_ne_pos_dict = {}

    for idx, t in enumerate(special_tokens + NE_LIST):
        ne_dict[t] = idx
        reverse_ne_dict[idx] = t

    for idx, t in enumerate(special_tokens + POS_LIST):
        pos_dict[t] = idx
        reverse_pos_dict[idx] = t

    for idx, t in enumerate(special_tokens + NE_POS_LIST):
        ne_pos_dict[t] = idx
        reverse_ne_pos_dict[idx] = t


    save_dict(ne_dict, config.NE_DICT_PATH)
    save_dict(pos_dict, config.POS_DICT_PATH)
    save_dict(ne_pos_dict, config.NE_POS_DICT_PATH)
    save_dict(reverse_ne_dict, config.REVERSE_NE_DICT_PATH)
    save_dict(reverse_pos_dict, config.REVERSE_POS_DICT_PATH)
    save_dict(reverse_ne_pos_dict, config.REVERSE_NE_POS_DICT_PATH)



def split_sentence_info(sent_data):
    morphSTR = []
    neSTR = []
    posSTR = []
    for sd in sent_data:
        morphSTR.append("".join([m[0] for m in sd['sgmMorphs']]) + "\n")
        posSTR.append("\t".join([m[1] for m in sd['sgmMorphs']]) + "\n")
        neSTR.append("\t".join([m[-1] for m in sd['sgmMorphs']]) + "\n")
    return morphSTR, posSTR, neSTR


def process_text(files, config):
    file_sentence_writer = open(config.PROCESS_SENT_DATA_PATH, "w", encoding="utf8")
    file_POS_writer = open(config.PROCESS_POS_PATH, "w", encoding="utf8")
    file_NE_writer = open(config.PROCESS_NE_PATH, "w", encoding="utf8")

    for filename in tqdm(files, desc="Pre-processing the text data....."):
        morphSTR, posSTR, neSTR = split_sentence_info(readData(filename))

        file_sentence_writer.writelines([fix_text(line) for line in morphSTR])
        file_POS_writer.writelines([fix_text(line) for line in posSTR])
        file_NE_writer.writelines([fix_text(line) for line in neSTR])

    file_sentence_writer.close()
    file_POS_writer.close()
    file_NE_writer.close()



def train_byte_pair_encoding(vocab_size, config):
    token_dict = Counter()
    with open(config.PROCESS_SENT_DATA_PATH, 'r', encoding='utf8') as fr:
        for line in fr.readlines():
            line = line.replace('</m>', '</m> ')
            line = line.replace('</sp>', ' </sp> ')
            line = line.replace('  ', ' ')
            line = line.lower().strip()
            token_dict.update(line.lower().split())

    with open(config.BPE_TSV_PATH, 'w', newline='', encoding='utf8') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for word in token_dict:
            tsv_output.writerow([word, token_dict[word]])

    spmcmd = '--input={spm_input} --model_prefix={spm_model} --input_format=tsv --vocab_size={vocab_size} --user_defined_symbols=</m>,</sp> --hard_vocab_limit=false --model_type=bpe --pad_id=0 --unk_id=3 --bos_id=1 --eos_id=2 --pad_piece=<pad> --unk_piece=<unk>'.format(
        spm_input=config.BPE_TSV_PATH, spm_model=config.BPE_MODEL_PATH, vocab_size=config.VOCAB_SIZE)

    spm.SentencePieceTrainer.train(spmcmd)


def file_list(path, fList=[]):
    for dir_path, dirs, files in os.walk(path):
        for filename in files:
            fList.append(os.path.join(dir_path, filename))
    return fList

def train(config=None):
    from pprint import pprint
    process_text(file_list(config.CORPUS_DIR_PATH), config)

    if config.is_pretraining_mode:
        train_byte_pair_encoding(vocab_size=config.VOCAB_SIZE, config=config)
