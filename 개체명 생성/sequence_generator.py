from src.sample import SequenceGenerator
from utils.data_pipeline import input_fn
from configuration import Config
# import tensorflow as tf
import os
import glob


def seq_gen(config, dataset):
    sg = SequenceGenerator(config)
    sg.load_weights()

    cnt = 1
    doc_id = 1
    fout = open(os.path.join(config.GENERATE_TEXT_PATH, 'generate_large_{:05}.txt'.format(doc_id)), mode='w', encoding='utf8')
    for context in dataset:

        context_ = context[0][0][1:5], context[1][0][1:5], context[-1][0][1:5]
        cnt += 1

        v_context = [int(i) for i in context_[0].numpy()]
        n_context = [int(i) for i in context_[1].numpy()]
        p_context = [int(i) for i in context_[-1].numpy()]

        generated_seq, raw_sentence = sg.sample_sequence(context=(v_context, n_context, p_context),
                                           seq_len=config.max_seq_len,
                                           temperature=config.temperature,
                                           top_k=config.top_k,
                                           top_p=config.top_p,
                                           nucleus_sampling=config.nucleus_sampling)


        r = raw_sentence.replace('</m>', '')
        r = r.replace('</sp>', ' ')
        r = r.replace('  ', ' ')
        print('$'+raw_sentence.replace('  ', ' '), file=fout)
        print(';'+r, file=fout)
        print("\n".join(generated_seq), file=fout)
        print("\n", file=fout)

        # print("="*30)

        if cnt % 10 == 0:
            print(cnt)
            fout.close()
            doc_id += 1
            fout = open(os.path.join(config.GENERATE_TEXT_PATH, 'generate_large_{:05}.txt'.format(doc_id)), mode='w', encoding='utf8')

    fout.close()



if __name__ == "__main__":
    config = Config()
    tf_records = glob.glob((config.TF_RECORDS_PATH+"/*.tfrecord"))
    dataset = input_fn(tf_records, batch_size=1, shuffle=False)
    seq_gen(config, dataset)

        # print("="*10)
        # print("\n".join(generated_seq))
        # print("*"*10)
