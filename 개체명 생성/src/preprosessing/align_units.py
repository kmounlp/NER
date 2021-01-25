# -*-encoding: utf-8-*-
import sys
import os
# if os.path.dirname(os.path.abspath(os.path.dirname(__file__))) not in sys.path:
#     sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import sentencepiece as spm
from .prepare_utils import load_dict
from tqdm import tqdm
from pprint import pprint

def prep_sents(line):
    line = line.replace('</m>', '</m> ')
    line = line.replace('</sp>', ' </sp> ')
    line = line.replace('  ', ' ')
    return line.strip()

def align_bp_ne_pos(config):
    sequences = []
    seq_sb = []
    seq_ne = []
    seq_pos = []
    bpe = spm.SentencePieceProcessor()
    bpe.Load(config.BPE_MODEL_PATH + ".model")

    bos_id = bpe.bos_id()
    eos_id = bpe.eos_id()

    ne_dict = load_dict(config.NE_DICT_PATH)
    pos_dict = load_dict(config.POS_DICT_PATH)
    ne_pos_dict = load_dict(config.NE_POS_DICT_PATH)

    sents = list(map(prep_sents, open(config.PROCESS_SENT_DATA_PATH, 'r', encoding='utf8').readlines()))
    ne_sents = [ line.strip() for line in open(config.PROCESS_NE_PATH, 'r', encoding='utf8').readlines() ]
    pos_sents = [ line.strip() for line in open(config.PROCESS_POS_PATH, 'r', encoding='utf8').readlines()]

    for s_sent, n_sent, p_sent in tqdm(list(zip(sents, ne_sents, pos_sents)), desc="Alinging multiple unit..."):
        # input(list(zip(s_sent.split(), n_sent.split())))
        for s, n, p in zip(s_sent.split(), n_sent.split(), p_sent.split()):
            for idx, piece_info in enumerate(zip(bpe.EncodeAsPieces(s), bpe.EncodeAsIds(s))):
                piece, s_id = piece_info
                if n == 'O':
                    n = p
                elif (idx != 0) and (n[0] == 'B'):
                    n = 'I' + n[1:]

                if (piece, p) == ('▁', '</sp>'): continue
                if piece == '</m>':
                    n, p = '</m>', '</m>'

                seq_sb.append(s_id)
                seq_ne.append(ne_pos_dict[n])
                seq_pos.append(pos_dict[p])
                # print(piece, s_id, n, ne_pos_dict[n], p, pos_dict[p])
                # input("="*20)

        sequences.append((seq_sb, seq_ne, seq_pos))
        seq_sb = []
        seq_ne = []
        seq_pos = []

    return sequences