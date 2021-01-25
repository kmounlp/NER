import os

class Config:

    def __init__(self):

        _ROOT = os.path.abspath(os.path.dirname(__file__))

        self.use_BIT = True
        self.is_pretraining_mode = True
        self.trainable = True

        self.num_layers = 4
        self.num_heads = 8
        self.d_model = 512
        self.n_model = 64
        self.p_model = 32
        self.dff = (self.d_model + self.n_model + self.p_model) * self.num_heads
        self.max_seq_len = 512
        self.batch_size = 4
        self.epoch = 50

        self.VOCAB_SIZE = 20000
        self.NE_SIZE = None
        self.POS_SIZE = None

        self.concat_d_model = self.d_model + self.n_model + self.p_model
        self.compress_d_model_1 = self.concat_d_model // 3
        self.compress_d_model_2 = self.compress_d_model_1 - (self.compress_d_model_1 % self.num_heads)

        self.learning_rate = 1e-5
        self.optimizer = "adam"

        self.temperature = 0.8
        self.top_k = 1
        self.top_p = 1
        self.nucleus_sampling = True,

        self.special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>', '</m>', '</sp>']

        LOG_DIR_NAME = 'add_log_kmouNER'
        MODEL_DIR_NAME = 'add_model_kmouNER'

        DATA_DIR_NAME = "data"
        CORPUS_DIR_NAME = "corpus/kmouNER"

        BPE_DIR_NAME = "add_kmou_make_bpe"
        DICT_DIR_NAME = "add_kmou_dict"

        BPE_TSV_NAME = "bpe_spm.tsv"
        BPE_MODEL_NAME = "add_kmou_spm"

        PREPROCESS_SENT_FILE = "sentences.txt"
        PREPROCESS_POS_SEQ_FILE = "POS_sentences.txt"
        PREPROCESS_NE_SEQ_FILE = "NE_sentences.txt"

        REVERSE_NE_DICT_NAME = 'r_ne.dict'
        REVERSE_POS_DICT_NAME = 'r_pos.dict'
        REVERSE_NE_POS_DICT_NAME = 'r_ne_pos.dict'

        NE_DICT_NAME = 'ne.dict'
        POS_DICT_NAME = 'pos.dict'
        NE_POS_DICT_NAME = 'ne_pos.dict'


        TF_RECORDS_DIR_NAME = 'tf_records_kmouNER'

        GENERATE_TEXT_DIR = 'no_`generate_kmouNER'
        self.GENERATE_TEXT_PATH = os.path.join(_ROOT, GENERATE_TEXT_DIR)

        DATA_DIR_PATH =  os.path.join(_ROOT, DATA_DIR_NAME)
        self.CORPUS_DIR_PATH = os.path.join(DATA_DIR_PATH, CORPUS_DIR_NAME)

        PROCESS_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, BPE_DIR_NAME)
        self.PROCESS_SENT_DATA_PATH = os.path.join(PROCESS_DATA_DIR_PATH, PREPROCESS_SENT_FILE)
        self.PROCESS_POS_PATH = os.path.join(PROCESS_DATA_DIR_PATH, PREPROCESS_POS_SEQ_FILE)
        self.PROCESS_NE_PATH = os.path.join(PROCESS_DATA_DIR_PATH, PREPROCESS_NE_SEQ_FILE)

        self.BPE_TSV_PATH = os.path.join(PROCESS_DATA_DIR_PATH, BPE_TSV_NAME)
        BPE_MODEL_DIR = os.path.join(_ROOT, "bin", BPE_MODEL_NAME)
        self.BPE_MODEL_PATH = os.path.join(BPE_MODEL_DIR, BPE_MODEL_NAME)
        DICT_PATH = os.path.join(_ROOT, "bin", DICT_DIR_NAME)

        self.REVERSE_NE_DICT_PATH = os.path.join(DICT_PATH, REVERSE_NE_DICT_NAME)
        self.REVERSE_POS_DICT_PATH = os.path.join(DICT_PATH, REVERSE_POS_DICT_NAME)
        self.REVERSE_NE_POS_DICT_PATH = os.path.join(DICT_PATH, REVERSE_NE_POS_DICT_NAME)

        self.NE_DICT_PATH = os.path.join(DICT_PATH, NE_DICT_NAME)
        self.POS_DICT_PATH = os.path.join(DICT_PATH, POS_DICT_NAME)
        self.NE_POS_DICT_PATH = os.path.join(DICT_PATH, NE_POS_DICT_NAME)

        self.TF_RECORDS_PATH = os.path.join(DATA_DIR_PATH, 'tf_records', TF_RECORDS_DIR_NAME)

        self.LOG_DIR =  os.path.join(_ROOT, LOG_DIR_NAME)
        self.MODEL_DIR =  os.path.join(_ROOT, MODEL_DIR_NAME)


        if not os.path.exists(DATA_DIR_PATH):
            os.makedirs(DATA_DIR_PATH)

        if not os.path.exists(PROCESS_DATA_DIR_PATH):
            os.makedirs(PROCESS_DATA_DIR_PATH)

        if not os.path.exists(PROCESS_DATA_DIR_PATH):
            os.makedirs(PROCESS_DATA_DIR_PATH)

        if not os.path.exists(BPE_MODEL_DIR):
            os.makedirs(BPE_MODEL_DIR)

        if not os.path.exists(DICT_PATH):
            os.makedirs(DICT_PATH)

        if not os.path.exists(self.TF_RECORDS_PATH):
            os.makedirs(self.TF_RECORDS_PATH)

        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)

        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)

        if not os.path.exists(os.path.join(self.LOG_DIR, 'plugins', 'profile')):
            os.makedirs(os.path.join(self.LOG_DIR, 'plugins', 'profile'))

        if not os.path.exists(self.GENERATE_TEXT_PATH):
            os.makedirs(self.GENERATE_TEXT_PATH)

