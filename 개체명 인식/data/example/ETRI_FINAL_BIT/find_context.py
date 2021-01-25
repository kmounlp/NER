import unicodedata
from collections import Counter
from pathlib import Path
from pprint import pprint

def tags(name):
    return '{}.tags.txt'.format(name)


print('Build vocab tags (may take a while)')
vocab_tags = set()
count = {
    "B_LC":{}, "B_OG" : {}, "B_PS": {}, "B_DT": {}, "B_TI": {}, "I_DT": {}, "I_TI":{}, "I_LC":{},"I_PS":{},"I_OG":{}

}
count2 = {
    "LC":{}, "OG" : {}, "PS": {}, "DT": {}, "TI": {}

}
for n in ['train', 'testa', 'testb']:
    with Path(tags(n)).open(encoding="utf8") as f:
        for line in f:
            line_list = line.split()
            for idx, ele in enumerate(line_list):
                if ele[1] == "_":
                    ele = ele[2:]
                    if idx == 0:
                        tag = line_list[idx+1]
                        if count2[ele].get(tag) is None:
                            count2[ele][tag] = 1
                        else:
                            count2[ele][tag] += 1
                        continue
                    elif idx == len(line_list) -1:
                        tag = line_list[idx-1]
                        if count2[ele].get(tag) is None:
                            count2[ele][tag] = 1
                        else:
                            count2[ele][tag] += 1
                        continue
                    else:
                        tag = line_list[idx+1]
                        if count2[ele].get(tag) is None:
                            count2[ele][tag] = 1
                        else:
                            count2[ele][tag] += 1
                        tag = line_list[idx-1]
                        if count2[ele].get(tag) is None:
                            count2[ele][tag] = 1
                        else:
                            count2[ele][tag] += 1

# pprint(count)
pprint(count2)
for ele in count2.keys():
    print(ele)
    print(len(ele))