from src.layers.feed_forward import *
from src.layers.attention_layer import *
from src.layers.embedding_layer import *
from tensorflow.keras.layers import Concatenate
from src.layers.layer_norm import LayerNormalization
import numpy as np
from tensorflow.python.framework import tensor_shape
import tensorflow_addons as tfa
import tensorflow as tf
from tqdm import tqdm
from pprint import pprint
import sys

from utils.tf_utils import *
import os

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="inputs_vocab"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="inputs_pos"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="inputs_ne"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="targets_vocab"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="targets_pos"),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32, name="targets_ne"),
    tf.TensorSpec(shape=(None), dtype=tf.int32, name="Step")
]


class Gpt2(tf.keras.Model):
    def __init__(self, config):
        super(Gpt2, self).__init__()

        self.LOG_DIR = config.LOG_DIR

        # self.rev_embedding_projection = rev_embedding_projection
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dff = config.dff
        self.max_seq_len = config.max_seq_len

        self.vocab_size = config.VOCAB_SIZE
        self.NE_size = config.NE_SIZE
        self.POS_size = config.POS_SIZE

        self.d_model = config.d_model
        self.n_model = config.n_model
        self.p_model = config.p_model

        self.concat_d_model = config.concat_d_model
        self.compress_d_model = config.compress_d_model_2

        self.learning_rate = config.learning_rate
        self.optimizer_t = config.optimizer
        self.dataset = None
        self.mirrored_strategy = None

        self.vocab_embedding = EmbeddingLayer(
            self.vocab_size, self.d_model)

        self.NE_embedding = EmbeddingLayer(
            self.NE_size, self.n_model)

        self.POS_embedding = EmbeddingLayer(
            self.POS_size, self.p_model)

        self.position_embedding = PositionEmbeddingLayer(
            self.max_seq_len, self.d_model)

        self.NE_position_embedding = PositionEmbeddingLayer(
            self.max_seq_len, self.n_model)

        self.POS_position_embedding = PositionEmbeddingLayer(
            self.max_seq_len, self.p_model)

        self.decoder_layers = [DecoderLayer(self.concat_d_model, self.num_heads, self.dff)
                               for _ in range(self.num_layers)]

        self.layer_norm = LayerNormalization(self.concat_d_model, name="hidden_layer_norm")

        self.layer_norm_sub = LayerNormalization(self.compress_d_model, name="vocab_layer_norm")
        self.layer_norm_pos = LayerNormalization(self.compress_d_model, name="POS_layer_norm")
        self.layer_norm_ne = LayerNormalization(self.compress_d_model, name="NE_layer_norm")

        ## candidate
        self.vocab_candidate_layer = OutputLayer(self.compress_d_model)
        self.POS_candidate_layer = OutputLayer(self.compress_d_model)
        self.NE_candidate_layer = OutputLayer(self.compress_d_model)

        # self.vocab_candidate_layer = OutputLayer(self.vocab_size)
        # self.POS_candidate_layer = OutputLayer(self.POS_size)
        # self.NE_candidate_layer = OutputLayer(self.NE_size)
        #
        # # ## relation
        # self.relation_layer = RelationLayer(self.compress_d_model, num_heads=config.num_heads)
        self.vocab_POS_layer = OutputLayer(self.d_model, name='vocab_POS_relation_2')
        self.vocab_NE_layer = OutputLayer(self.d_model, name='vocab_NE_relation_2')
        self.POS_NE_layer = OutputLayer(self.d_model, name='POS_NE_relation_2')

        self.vocab_relation = OutputLayer(self.d_model, name='vocab_POS_relation_3')
        self.POS_relation = OutputLayer(self.d_model, name='vocab_NE_relation_3')
        self.NE_relation = OutputLayer(self.d_model, name='POS_NE_relation_3')

        ## output
        self.vocab_output_layer = OutputLayer(self.vocab_size)
        self.NE_output_layer = OutputLayer(self.NE_size)
        self.POS_output_layer = OutputLayer(self.POS_size)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                         reduction=tf.keras.losses.Reduction.NONE)
        self.accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_object')

        self.train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int32)]


    def call(self, x, training=True, past=None):
        vocab_x, POS_x, NE_x = x

        vocab_x = tf.cast(vocab_x, tf.int32)
        POS_x = tf.cast(POS_x, tf.int32)
        NE_x = tf.cast(NE_x, tf.int32)

        # self.batch_size, self.sequence = tf.shape(x)[0], tf.shape(x)[1]
        if past is None:
            pasts = [None] * self.num_layers
        else:
            pasts = past

        assert len(pasts) == self.num_layers

        att_mask = create_masks(vocab_x)

        batch, sequence = tf.shape(vocab_x)[0], tf.shape(vocab_x)[1]

        past_length = 1 if past is None else tf.shape(past)[-2]

        with tf.name_scope("embeddings"):
            vocab_embedded_x = self.vocab_embedding(vocab_x) + self.position_embedding(vocab_x, start=past_length)
            POS_embedded_x = self.POS_embedding(POS_x) + self.POS_position_embedding(POS_x, start=past_length)
            NE_embedded_x = self.NE_embedding(NE_x) + self.NE_position_embedding(NE_x, start=past_length)

            hidden_states = tf.concat([vocab_embedded_x, POS_embedded_x, NE_embedded_x], axis=-1)

        presents = []

        for decoder_layer, past in zip(self.decoder_layers, pasts):
            hidden_states, present = decoder_layer(hidden_states, training, att_mask, past=past)
            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)

        vocab_hidden_states = self.vocab_candidate_layer(hidden_states)
        POS_hidden_states = self.POS_candidate_layer(hidden_states)
        NE_hidden_states = self.NE_candidate_layer(hidden_states)

        vocab_hidden_states = self.layer_norm_sub(vocab_hidden_states)
        POS_hidden_states = self.layer_norm_pos(POS_hidden_states)
        NE_hidden_states = self.layer_norm_ne(NE_hidden_states)

        vp_states = self.vocab_POS_layer(tf.concat([vocab_hidden_states, POS_hidden_states], axis=-1))
        vn_states = self.vocab_NE_layer(tf.concat([vocab_hidden_states, NE_hidden_states], axis=-1))
        pn_states = self.POS_NE_layer(tf.concat([POS_hidden_states, NE_hidden_states], axis=-1))

        v_relation = self.vocab_relation(tf.concat([vp_states, vn_states], axis=-1))
        p_relation = self.POS_relation(tf.concat([vp_states, pn_states], axis=-1))
        n_relation = self.NE_relation(tf.concat([vn_states, pn_states], axis=-1))

        # vocab_logits = self.vocab_output_layer(hidden_states)
        # POS_logits = self.POS_output_layer(hidden_states)
        # NE_logits = self.NE_output_layer(hidden_states)

        vocab_logits = self.vocab_output_layer(tf.concat([vocab_hidden_states, v_relation], axis=-1))
        POS_logits = self.POS_output_layer(tf.concat([POS_hidden_states, p_relation], axis=-1))
        NE_logits = self.NE_output_layer(tf.concat([NE_hidden_states, n_relation], axis=-1))

        # vocab_logits = self.layer_norm_sub(vocab_logits)
        # POS_logits = self.layer_norm_pos(POS_logits)
        # NE_logits = self.layer_norm_ne(NE_logits)

        # vocab_logits = tf.nn.softmax(vocab_logits)
        # POS_logits = tf.nn.softmax(POS_logits)
        # NE_logits = tf.nn.softmax(NE_logits)

        return (vocab_logits, POS_logits, NE_logits), presents

    @staticmethod
    def get_padded_accuracy(labels, logits):
        with tf.name_scope("padded_accuracy"):
            weights = tf.cast(tf.not_equal(labels, 0), tf.float32)

            outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
            padded_labels = tf.cast(labels, tf.int32)

            nonpad_seq = tf.math.count_nonzero(weights, dtype=tf.dtypes.float32, )
            acc = tf.cast(tf.equal(outputs, padded_labels), tf.float32)

            accuracy = tf.reduce_sum(tf.cast(acc * weights, tf.float32)) / nonpad_seq
            return tf.cast(accuracy, tf.float32)

    def create_optimizer(self):
        optimizer = self.optimizer_t.lower()

        with tf.name_scope("optimizer"):
            if optimizer == "radam":
                self.global_optimizer = tfa.optimizers.RectifiedAdam(lr=self.learning_rate)
                # self.vocab_optimizer = tfa.optimizers.RectifiedAdam(lr=self.learning_rate)
                # self.ne_optimizer = tfa.optimizers.RectifiedAdam(lr=self.learning_rate)
                # self.pos_optimizer = tfa.optimizers.RectifiedAdam(lr=self.learning_rate)

            elif optimizer == "adam":
                self.global_optimizer = tf.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                                                           epsilon=1e-9)
                # self.vocab_optimizer = tf.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                #                                           epsilon=1e-9)
                # self.ne_optimizer = tf.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                #                                        epsilon=1e-9)
                # self.pos_optimizer = tf.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98,
                #                                         epsilon=1e-9)

            elif optimizer == "adadelta":
                self.global_optimizer = tf.optimizers.Adadelta(self.learning_rate)
                # self.vocab_optimizer = tf.optimizers.Adadelta(self.learning_rate)
                # self.ne_optimizer = tf.optimizers.Adadelta(self.learning_rate)
                # self.pos_optimizer = tf.optimizers.Adadelta(self.learning_rate)

            elif optimizer == "rms":
                self.global_optimizer = tf.optimizers.RMSprop(self.learning_rate)
                # self.vocab_optimizer = tf.optimizers.RMSprop(self.learning_rate)
                # self.ne_optimizer = tf.optimizers.RMSprop(self.learning_rate)
                # self.pos_optimizer = tf.optimizers.RMSprop(self.learning_rate)

            else:
                self.global_optimizer = tf.optimizers.SGD(self.learning_rate)
                # self.vocab_optimizer = tf.optimizers.SGD(self.learning_rate)
                # self.ne_optimizer = tf.optimizers.SGD(self.learning_rate)
                # self.pos_optimizer = tf.optimizers.SGD(self.learning_rate)

            return self.global_optimizer

            # return self.global_optimizer, self.vocab_optimizer, self.ne_optimizer, self.pos_optimizer


    @tf.function
    def get_loss(self, real, pred, loss_type=None):
        with tf.name_scope("loss_layer"):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = self.loss_object(real, pred)

            if loss_type == 'NE':
                tf.print(tf.math.is_nan(pred))
                print("-"*20)
                tf.print(tf.math.is_nan(real))
                print("="*20)
                tf.print(tf.argmax(pred))
                print("-" * 20)
                tf.print(real)
                print("*"*20)

            with tf.name_scope("loss_masking"):
                mask = tf.cast(mask, dtype=loss_.dtype)
                loss_ *= mask
            loss_ = tf.reduce_sum(loss_, axis=1)
            sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)

            return sequence_avg_loss


    @staticmethod
    def get_perplexity(cross_entropy):
        perplexity = tf.exp(cross_entropy)
        return perplexity

    def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
        with tf.name_scope('checkpoint_manager'):
            ckpt = tf.train.Checkpoint(global_optimizer=self.global_optimizer, model=self)
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

            if load_model:  # If want to load trained weights
                ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored...............')
            else:
                print("Initializing model from scratch..........")

    def load_model(self, filepath):
        ckpt = tf.train.Checkpoint(model=self)
        ckpt_manager = tf.train.CheckpointManager(ckpt, filepath)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Model Restored..........................")

    def create_summary_writer(self, summary_path):
        train_summary_path = summary_path + "/train"
        test_summary_path = summary_path + "/test"

        with tf.name_scope('summary'):
            self.train_writer = tf.summary.create_file_writer(train_summary_path)
            self.test_writer = tf.summary.create_file_writer(test_summary_path)

            return self.train_writer, self.test_writer

    def _train_step(self, inputs_vocab, inputs_pos, inputs_ne, targets_vocab, targets_pos, targets_ne, step,
                   grad_clip=True, clip_value_min=-1.0, clip_value_max=1.0):

        inputs = inputs_vocab, inputs_pos, inputs_ne
        targets = targets_vocab, targets_pos, targets_ne

        with tf.GradientTape(persistent=True) as tape2:
            predictions, _ = self(inputs, training=True)

            vocab_loss = tf.reduce_mean(self.get_loss(targets[0], predictions[0], "vocab"))
            POS_loss = tf.reduce_mean(self.get_loss(targets[1], predictions[1], "pos"))
            NE_loss = tf.reduce_mean(self.get_loss(targets[-1], predictions[-1], "ne"))

            loss = vocab_loss + POS_loss + NE_loss
            #
            # if float(NE_loss) == float('inf'):
            #     print("?")
            #     ne_out = open('NE_loss_nan.txt', mode='a', encoding='utf8')
            #     print(int(step), NE_loss, sep='\t', file=ne_out)
            #     ne_out.close()
            #
            # if float(POS_loss) == float('inf'):
            #     pos_out = open('POS_loss_nan.txt', mode='a', encoding='utf8')
            #     print(int(step), POS_loss, sep='\t', file=pos_out)
            #     pos_out.close()

        with tf.name_scope("joint_gradients"):
            joint_gradients = tape2.gradient(loss, self.trainable_variables)
            if grad_clip:
                joint_gradients = [(tf.clip_by_value(grad, -clip_value_min, clip_value_max)) for grad in
                                   joint_gradients]
            self.global_optimizer.apply_gradients(zip(joint_gradients, self.trainable_variables))


        vocab_accuracy = self.get_padded_accuracy(targets[0], predictions[0])
        POS_accuracy = self.get_padded_accuracy(targets[1], predictions[1])
        NE_accuracy = self.get_padded_accuracy(targets[-1], predictions[-1])

        vocab_accuracy = tf.convert_to_tensor(vocab_accuracy, np.float32, name="v_acc")
        NE_accuracy = tf.convert_to_tensor(NE_accuracy, np.float32, name="n_acc")
        POS_accuracy = tf.convert_to_tensor(POS_accuracy, np.float32, name="p_acc")

        harmonic_accuracy = (vocab_accuracy + NE_accuracy + POS_accuracy) / 3

        vocab_perplexity = self.get_perplexity(vocab_loss)
        NE_perplexity = self.get_perplexity(NE_loss)
        POS_perplexity = self.get_perplexity(POS_loss)

        perplexity = (vocab_perplexity + NE_perplexity + POS_perplexity) / 3

        loss_ = {'loss': loss, 'vocab_loss': vocab_loss, 'NE_loss': NE_loss, 'POS_loss': POS_loss,
                 'perplexity':perplexity, 'vocab_perplexity': vocab_perplexity,  'NE_perplexity' : NE_perplexity, 'POS_perplexity' : POS_perplexity}
        accuracy_ = {'accuracy': harmonic_accuracy, 'vocab_accuracy': vocab_accuracy, 'NE_accuracy': NE_accuracy,
                     'POS_accuracy': POS_accuracy}

        self.log_summary(self.train_writer,
                         step,
                         loss_,
                         accuracy_)

        return loss_, accuracy_, predictions

    def _test_step(self, test_inputs_vocab, test_inputs_ne, test_inputs_pos, test_targets_vocab, test_targets_ne, test_targets_pos, dummy=False):
        inputs = test_inputs_vocab, test_inputs_pos, test_inputs_ne,
        targets = test_targets_vocab, test_targets_pos, test_targets_ne

        pred, _ = self(inputs, training=False)
        vocab_loss = self.get_loss(targets[0], pred[0], 'SUBWORD')
        POS_loss = self.get_loss(targets[1], pred[1], 'POS')
        NE_loss = self.get_loss(targets[-1], pred[-1], 'POS')

        vocab_perplexity = self.get_perplexity(vocab_loss)
        NE_perplexity = self.get_perplexity(NE_loss)
        POS_perplexity = self.get_perplexity(POS_loss)

        loss = (vocab_loss + POS_loss + NE_loss) / 3
        perplexity = (vocab_perplexity + NE_perplexity + POS_perplexity) / 3

        vocab_accuracy = self.get_padded_accuracy(targets[0], pred[0])
        POS_accuracy = self.get_padded_accuracy(targets[1], pred[1])
        NE_accuracy = self.get_padded_accuracy(targets[-1], pred[-1])

        vocab_accuracy = tf.convert_to_tensor(vocab_accuracy, np.float32, name="v_acc")
        NE_accuracy = tf.convert_to_tensor(NE_accuracy, np.float32, name="n_acc")
        POS_accuracy = tf.convert_to_tensor(POS_accuracy, np.float32, name="p_acc")

        harmonic_accuracy = (vocab_accuracy + NE_accuracy + POS_accuracy) / 3

        loss_ = {'loss': loss, 'vocab_loss': vocab_loss, 'NE_loss': NE_loss, 'POS_loss': POS_loss,
                 'perplexity': perplexity, 'vocab_perplexity': vocab_perplexity,  'NE_perplexity' : NE_perplexity, 'POS_perplexity' : POS_perplexity}

        accuracy_ = {'accuracy': harmonic_accuracy, 'vocab_accuracy': vocab_accuracy, 'NE_accuracy': NE_accuracy,
                     'POS_accuracy': POS_accuracy}

        return loss_, accuracy_, pred

    @tf.function(input_signature=train_step_signature)
    def train_step(self,  inputs_vocab, inputs_pos, inputs_ne, targets_vocab, targets_pos, targets_ne, step, grad_clip=True, clip_value_min=2.5, clip_value_max=2.5):
        return self._train_step( inputs_vocab, inputs_pos, inputs_ne, targets_vocab, targets_pos, targets_ne, step, grad_clip, clip_value_min, clip_value_max)

    @tf.function(input_signature=train_step_signature)
    def test_step(self, test_inputs_vocab, test_inputs_pos, test_inputs_ne, test_targets_vocab, test_targets_pos, test_targets_ne, dummy=False):
        return self._test_step(test_inputs_vocab, test_inputs_pos, test_inputs_ne, test_targets_vocab, test_targets_pos, test_targets_ne, dummy=False)

    def get_train_test_function(self, graph_mode=False):
        # if graph_mode:
        print("Running in graph mode.............")
        train_fuc = self.train_step
        test_fuc = self.test_step
        # else:
        #     print("Running in eager mode.............")
        #     train_fuc = self._train_step
        #     test_fuc = self._test_step
        return train_fuc, test_fuc

    def fit(self, dataset, graph_mode=True):
        train_func, test_func = self.get_train_test_function(graph_mode)
        tf.summary.trace_on(graph=True, profiler=False)
        train_dataset, test_dataset = dataset
        train_dataset = tqdm(list(enumerate(train_dataset)))

        for step, (inputs_vocab, inputs_pos, inputs_ne, targets_vocab, targets_pos, targets_ne) in train_dataset:
            train_loss, train_acc, train_pred = train_func(inputs_vocab, inputs_pos, inputs_ne, targets_vocab, targets_pos, targets_ne, step)

            # if step % 10 == 0:
            train_dataset.set_description('v : %.04f n : %0.4f p : %0.4f' % (
                train_loss['vocab_loss'], train_loss['NE_loss'], train_loss['POS_loss']))
            train_dataset.set_postfix({'step': step,
                                       'Joint_Loss': train_loss['loss'].numpy(),
                                       'ACC': train_acc['accuracy'].numpy()}, refresh=True)

            if step % 1000 == 0:
                losses = []
                vocab_losses = []
                NE_losses = []
                POS_losses = []

                perplexities = []
                vocab_perplexities = []
                NE_perplexities = []
                POS_perplexities = []

                test_acc_ = []
                test_vocab_acc_ = []
                test_NE_acc_ = []
                test_POS_acc_ = []

                for (test_step, (test_inputs_vocab, test_inputs_pos, test_inputs_ne, test_targets_vocab, test_targets_pos, test_targets_ne)) in enumerate(test_dataset):
                    test_loss, test_accuracy, test_pred = test_func(test_inputs_vocab, test_inputs_pos, test_inputs_ne, test_targets_vocab, test_targets_pos, test_targets_ne)
                    losses.append(test_loss['loss'])
                    vocab_losses.append(test_loss['vocab_loss'])
                    NE_losses.append(test_loss['NE_loss'])
                    POS_losses.append(test_loss['POS_loss'])

                    perplexities.append(test_loss['perplexity'])
                    vocab_perplexities.append(test_loss['vocab_perplexity'])
                    NE_perplexities.append(test_loss['NE_perplexity'])
                    POS_perplexities.append(test_loss['POS_perplexity'])

                    test_acc_.append(test_accuracy['accuracy'])
                    test_vocab_acc_.append(test_accuracy['vocab_accuracy'])
                    test_NE_acc_.append(test_accuracy['NE_accuracy'])
                    test_POS_acc_.append(test_accuracy['POS_accuracy'])

                    if test_step == 100:
                        break

                test_loss = {'loss': np.mean(np.array(losses)), 'vocab_loss': np.mean(np.array(vocab_losses)),
                             'NE_loss': np.mean(np.array(NE_losses)), 'POS_loss': np.mean(np.array(POS_losses)),
                             'perplexity': np.mean(np.array(perplexities)),
                             'vocab_perplexity': np.mean(np.array(vocab_perplexities)),
                             'NE_perplexity': np.mean(np.array(NE_perplexities)),
                             'POS_perplexity': np.mean(np.array(POS_perplexities))}

                test_accuracy = {'accuracy': np.mean(np.array(test_acc_)),
                                 'vocab_accuracy': np.mean(np.array(test_vocab_acc_)),
                                 'NE_accuracy': np.mean(np.array(test_NE_acc_)),
                                 'POS_accuracy': np.mean(np.array(test_POS_acc_))}

                self.log_summary(self.test_writer,
                                 step,
                                 test_loss,
                                 test_accuracy,
                                 result_type="test")

                self.ckpt_manager.save()

            if step == 0:
                with self.train_writer.as_default():
                    tf.summary.trace_export(
                        name="gpt-2",
                        step=0,
                        profiler_outdir=self.LOG_DIR)

            if step % 1000 == 0:
                try:
                    self.ckpt_manager.save()
                except UnicodeDecodeError:
                    train_dataset.set_description("UnicodeDecodeError")

    @staticmethod
    def log_summary(tf_writer, step, loss_, accuracy_, result_type="Train"):
        with tf.name_scope("summary_writer"):
            with tf_writer.as_default():
                tf.summary.scalar("loss", loss_['loss'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("vocab_loss", loss_['vocab_loss'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("NE_loss", loss_['NE_loss'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("POS_loss", loss_['POS_loss'], step=tf.cast(step, tf.int64))

                tf.summary.scalar("perplexity", loss_['perplexity'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("vocab_perplexity", loss_['vocab_perplexity'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("NE_perplexity", loss_['NE_perplexity'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("POS_perpelxity", loss_['POS_perplexity'], step=tf.cast(step, tf.int64))
                #
                tf.summary.scalar("accuracy", accuracy_['accuracy'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("vocab_accuracy", accuracy_['vocab_accuracy'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("NE_accuracy", accuracy_['NE_accuracy'], step=tf.cast(step, tf.int64))
                tf.summary.scalar("POS_accuracy", accuracy_['POS_accuracy'], step=tf.cast(step, tf.int64))


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, name="", proj_weights=None, kernel_initializer=None):
        super(OutputLayer, self).__init__()
        self.proj_weights = proj_weights
        self.output_dim = output_dim
        self.layer_weights = None
        self.kernel_initializer = kernel_initializer
        self._name = name

    def build(self, input_shape):
        if self.proj_weights is None:
            input_dim = tensor_shape.dimension_value(input_shape[-1])
            self.layer_weights = self.add_weight(
                'output_layer_weights',
                shape=[input_dim, self.output_dim],
                initializer=self.kernel_initializer,
                trainable=True)
        super(OutputLayer, self).build(input_shape)

    def call(self, x):
        batch, sequence, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1]
        h_flat = tf.reshape(x, [-1, d_model])

        if self.proj_weights is None:
            out = tf.matmul(h_flat, self.layer_weights)
        else:
            out = tf.matmul(h_flat, self.porj_weights, transpose_b=True)
        out = tf.reshape(out, [batch, sequence, self.output_dim])
        return out
#
# class RelationLayer(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, dr_rate=0.1):
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.dff = self.d_model * 2 * self.num_heads
#         self.dr_rate = dr_rate
#
#         self.vocab_POS_layer = OutputLayer(self.d_model, name='vocab_POS_relation_2')
#         self.vocab_NE_layer = OutputLayer(self.d_model, name='vocab_NE_relation_2')
#         self.POS_NE_layer = OutputLayer(self.d_model, name='POS_NE_relation_2')
#
#         # self.vocab_relation_layer = OutputLayer(self.d_model, name='vocab_relation')
#         # self.POS_relation_layer = OutputLayer(self.d_model, name='POS_relation')
#         # self.NE_relation_layer = OutputLayer(self.d_model, name='NE_relation')
#
#         self.vocab_relation = MultiHeadAttention(self.d_model*2, self.num_heads)
#         self.vocab_feed_forward = FeedForward(self.d_model*2, self.dff, self.dr_rate)
#         self.vocab_layer_norm1 = LayerNormalization(self.d_model*2)
#         self.vocab_layer_norm2 = LayerNormalization(self.d_model*2)
#
#         self.POS_relation = MultiHeadAttention(self.d_model*2, self.num_heads)
#         self.POS_feed_forward = FeedForward(self.d_model*2, self.dff, self.dr_rate)
#         self.POS_layer_norm1 = LayerNormalization(self.d_model*2)
#         self.POS_layer_norm2 = LayerNormalization(self.d_model*2)
#
#         self.NE_relation = MultiHeadAttention(self.d_model*2, self.num_heads)
#         self.NE_feed_forward = FeedForward(self.d_model*2, self.dff, self.dr_rate)
#         self.NE_layer_norm1 = LayerNormalization(self.d_model*2)
#         self.NE_layer_norm2 = LayerNormalization(self.d_model*2)
#
#         # self.vocab_relation_attention_seq = tf.keras.layers.Attention()
#         # self.POS_relation_attention_seq = tf.keras.layers.Attention()
#         # self.NE_relation_layer_seq = tf.keras.layers.Attention()
#         #
#         # self.vocab_encoding_layer = tf.keras.layers.GlobalAveragePooling1D()
#         # self.POS_encoding_layer = tf.keras.layers.GlobalAveragePooling1D()
#         # self.NE_encoding_layer = tf.keras.layers.GlobalAveragePooling1D()
#         #
#         # self.vocab_relation_score_layer = tf.keras.layers.GlobalAveragePooling1D()
#         # self.POS_relation_score_layer = tf.keras.layers.GlobalAveragePooling1D()
#         # self.NE_relation_score_layer = tf.keras.layers.GlobalAveragePooling1D()
#
#     def call(self, c_vocab, c_POS, c_NE):
#         with tf.name_scope("relation_generation_unit"):
#             vocab_POS = self.vocab_POS_layer(tf.concat([c_vocab, c_POS], axis=-1))
#             vocab_NE = self.vocab_NE_layer(tf.concat([c_vocab, c_NE], axis=-1))
#             POS_NE = self.POS_NE_layer(tf.concat([c_POS, c_NE], axis=-1))
#
#             vocab_x = tf.concat([vocab_POS, vocab_NE], axis=-1)
#             POS_x = tf.concat([vocab_POS, POS_NE], axis=-1)
#             NE_x = tf.concat([vocab_NE, POS_NE], axis=-1)
#
#             vocab_relation_attention, _ = self.vocab_r
#             elation(self.vocab_layer_norm1(vocab_x))
#             with tf.name_scope("residual_conn"):
#                 vocab_x = vocab_x + vocab_relation_attention
#             vocab_relation_attention = self.vocab_feed_forward(self.vocab_layer_norm2(vocab_x))
#             with tf.name_scope("residual_conn"):
#                 vocab_x = vocab_x + vocab_relation_attention
#
#             POS_relation_attention, _ = self.POS_relation(self.POS_layer_norm1(POS_x))
#             with tf.name_scope("residual_conn"):
#                 POS_x = POS_x + POS_relation_attention
#             POS_relation_attention, _ = self.POS_feed_forward(self.POS_layer_norm2(POS_x))
#             with tf.name_scope("residual_conn"):
#                 POS_x = POS_x + POS_relation_attention
#
#             NE_relation_attention, _ = self.NE_relation(self.NE_layer_norm1(NE_x))
#             with tf.name_scope("residual_conn"):
#                 NE_x = NE_x + NE_relation_attention
#             POS_relation_attention, _ = self.NE_feed_forward(self.NE_layer_norm2(NE_x))
#             with tf.name_scope("residual_conn"):
#                 NE_x = NE_x + NE_relation_attention
#
#         return vocab_x, POS_x, NE_x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,
                 dr_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dr_rate = dr_rate

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.feed_forward = FeedForward(self.d_model, self.dff, self.dr_rate)
        self.layer_norm1 = LayerNormalization(self.d_model)
        self.layer_norm2 = LayerNormalization(self.d_model)

    def call(self, x, training, mask, past=None):
        out, present = self.mha(self.layer_norm1(x), mask=mask, past_layer=past,
                                training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        out = self.feed_forward(self.layer_norm2(x), training=training)  # (batch_size, input_seq_len, d_model)
        with tf.name_scope("residual_conn"):
            x = x + out
        return x, present
