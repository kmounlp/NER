from pprint import pprint
import itertools
import math

def pairwise(iterable: object) -> object:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    assert isinstance(b, object)
    return zip(a, b)


word_tag_count = dict()
tag_count = dict()
pre_tag_tag_count=dict()
with open('train.words.txt','r',encoding="utf8") as rf, open('train.tags.txt','r',encoding="utf8") as tf:
    for wline, tline in zip(rf,tf):
        w_list = wline.strip().split()
        t_list = tline.strip().split()
        if len(w_list) != len(t_list):
            print("_______________________________")
            continue
        for w,t in zip(w_list,t_list):
            w_t = w + "_" +t
            if word_tag_count.get(w_t) is None:
                word_tag_count[w_t] = 1
            else:
                word_tag_count[w_t] += 1
with open('train.tags.txt','r',encoding="utf8") as tf:
    for tline in tf:
        t_list = tline.strip().split()
        for t in t_list:
            if tag_count.get(t) is None:
                tag_count[t] =1
            else:
                tag_count[t] +=1
with open('train.tags.txt','r',encoding="utf8") as tf:
    for tline in tf:
        t_list = tline.strip().split()
        t_list.insert(0,"BOS")
        for pre_t,cur_t in pairwise(t_list):
            pre_cur = pre_t +"_"+cur_t
            if pre_tag_tag_count.get(pre_cur) is None:
                pre_tag_tag_count[pre_cur] =1
            else:
                pre_tag_tag_count[pre_cur] +=1
pprint(tag_count)
with open('train.words.txt','r',encoding="utf8") as rf, open('train.tags.txt','r',encoding="utf8") as tf:
    total = 0
    line_count =0
    for wline, tline in zip(rf, tf):
        sentence_per = 0

        w_list = wline.strip().split()
        t_list = tline.strip().split()
        if len(w_list) != len(t_list):
            print("_______________________________")
            continue
        for idx in range(len(w_list)):
            if idx == 0:
                w_t = w_list[idx] + "_" + t_list[idx]
                t = t_list[idx]
                pre_cur_tag = "BOS_"+t_list[idx]
            else:
                w_t = w_list[idx] + "_" + t_list[idx]
                t = t_list[idx]
                pre_cur_tag = t_list[idx-1] +"_"+t_list[idx]
            w_t_c = word_tag_count[w_t]
            t_c = tag_count[t]
            p_c_t = pre_tag_tag_count[pre_cur_tag]
            sentence_per += math.log(w_t_c/t_c)+math.log(t_c/p_c_t)
            line_count += 1
        total += sentence_per
    print(pow(2,-total/line_count))