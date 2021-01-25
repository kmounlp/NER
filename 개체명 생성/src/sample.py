import tensorflow as tf
import sentencepiece as spm
from .gpt_model import Gpt2
import json
import pickle
import os


def argmax(logits):
    return tf.argmax(logits)


def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1]

    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )


# Nucleas Sampling (https://arxiv.org/pdf/1904.09751.pdf)


def top_p_logits(logits, p):
    """Took from OpenAI GPT-2 Implememtation"""
    batch = tf.shape(logits)[0]
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


class SequenceGenerator:
    def __init__(self, config):
        self.sp = None
        self.model = None
        self.model_path = config.MODEL_DIR
        self.model_param = os.path.join(self.model_path, 'model_par.json')
        self.vocab_path = config.BPE_MODEL_PATH + '.model'
        self.config = config


        self.POS_DICT = pickle.load(open(config.POS_DICT_PATH, 'rb'))
        self.REVERSE_POS_DICT = pickle.load(open(config.REVERSE_POS_DICT_PATH, 'rb'))

        if config.use_BIT:
            self.NE_DICT = pickle.load(open(config.NE_POS_DICT_PATH, 'rb'))
            self.REVERSE_NE_DICT = pickle.load(open(config.REVERSE_NE_POS_DICT_PATH, 'rb'))
        else:
            self.NE_DICT = pickle.load(open(config.NE_DICT_PATH, 'rb'))
            self.REVERSE_NE_DICT = pickle.load(open(config.REVERSE_NE_DICT_PATH, 'rb'))


    # def __init__(self, model_path, model_param, vocab_path):
    #     self.sp = None
    #     self.model = None
    #     self.model_path = model_path
    #     self.model_param = model_param
    #     self.vocab_path = vocab_path

    def load_weights(self):

        with open(self.model_param) as f:
            param = json.load(f)

        self.config.num_layers = param['num_layers']
        self.config.d_model = param['d_model']
        self.config.n_model = param['n_model']
        self.config.p_model = param['p_model']
        self.config.num_heads = param['num_heads']
        self.config.dff = param['dff']
        self.config.max_seq_len = param['max_seq_len']
        self.config.VOCAB_SIZE = param['vocab_size']
        self.config.NE_SIZE = param['ne_size']
        self.config.POS_SIZE = param['pos_size']


        self.model = Gpt2(self.config)

        ckpt = tf.train.Checkpoint(model=self.model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, self.model_path, max_to_keep=1)

        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Model weights loaded into memory')

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.vocab_path)

        self.space_subword = self.sp.EncodeAsIds('</sp>')
        self.morph_subword = self.sp.EncodeAsIds('</m>')

        self.space_ne = self.REVERSE_NE_DICT.get('</sp>')
        self.morph_ne = self.REVERSE_NE_DICT.get('</m>')

        self.space_pos = self.REVERSE_POS_DICT.get('</sp>')
        self.morph_pos = self.REVERSE_POS_DICT.get('</m>')


    def sample_sequence(self,
                        context=None,
                        seq_len=512,
                        bos=1,
                        eos=2,
                        temperature=0.5,
                        top_k=1,
                        top_p=1,
                        nucleus_sampling=True):

        vocab_context, ne_context, pos_context = context

        if context == None:
            print("Give some context to model.................")
            return

        vocab_context, ne_context, pos_context = context

        vocab_context = tf.expand_dims(([bos] + vocab_context), 0)
        ne_context = tf.expand_dims(([bos] + ne_context), 0)
        pos_context = tf.expand_dims(([bos] + pos_context), 0)

        prev = vocab_context, ne_context, pos_context

        vocab_output = vocab_context
        ne_output = ne_context
        pos_output = pos_context

        past = None

        for i in range(seq_len):
            logits, past = self.model(prev, training=False, past=past)

            vocab_logits, ne_logits, pos_logits = logits
            vocab_logits = vocab_logits[:, -1, :] / tf.cast(temperature, tf.float32)
            ne_logits = ne_logits[:, -1, :] / tf.cast(temperature, tf.float32)
            pos_logits = pos_logits[:, -1, :] / tf.cast(temperature, tf.float32)

            vocab_logits = top_k_logits(vocab_logits, k=top_k)
            ne_logits = top_k_logits(ne_logits, k=top_k)
            pos_logits = top_k_logits(pos_logits, k=top_k)

            if nucleus_sampling:
                vocab_logits = top_p_logits(vocab_logits, p=top_p)
                ne_logits = top_p_logits(ne_logits, p=top_p)
                pos_logits = top_p_logits(pos_logits, p=top_p)


            samples_vocab = tf.random.categorical(vocab_logits, num_samples=1, dtype=tf.int32)
            samples_ne = tf.random.categorical(ne_logits, num_samples=1, dtype=tf.int32)
            samples_pos = tf.random.categorical(pos_logits, num_samples=1, dtype=tf.int32)

            if tf.equal(samples_vocab, eos):
                break

            vocab_output = tf.concat([vocab_output, samples_vocab], axis=-1)
            ne_output = tf.concat([ne_output, samples_ne], axis=-1)
            pos_output = tf.concat([pos_output, samples_pos], axis=-1)

            prev = samples_vocab, samples_ne, samples_pos

        vocab_result = tf.squeeze(vocab_output, axis=0)
        ne_result = tf.squeeze(ne_output, axis=0)
        pos_result = tf.squeeze(pos_output, axis=0)

        vocab_pred = [int(i) for i in vocab_result]
        ne_pred = [int(i) for i in ne_result]
        pos_pred = [int(i) for i in pos_result]

        generated_seq = []

        raw_sentence = self.sp.decode_ids(vocab_pred[1:])

        for v, n, p in zip(vocab_pred[1:], ne_pred[1:], pos_pred[1:]):
            generated_seq.append("{}\t{}\t{}".format(self.sp.IdToPiece(v), self.REVERSE_POS_DICT.get(p, '<unk>'), self.REVERSE_NE_DICT.get(n, '<unk>')))

        return generated_seq, raw_sentence
