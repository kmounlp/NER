import copy
import re
import string
from pprint import pprint

ne_compile = re.compile('<.+?:[A-Z]{2,3}>')

def is_alphabet(morph):
    for i in morph:
        if i not in string.ascii_letters:
            return False
    return True

def decision_sent_info(line, sent_info):
    if line[0] == '#': ## sentence source info
        return
    if line[0] == ';':  ## raw sentence
        sent_info['rawSTR'] = line[1:].strip()
    elif line[0] == '$':  ## ne sentence
        sent_info['neSTR/Dummy'] = line[1:].strip()
        sent_info['tokens'] = []

        if ne_compile.findall(line[1:].strip()):
            sent_info['existNE'] = True
        else:
            sent_info['existNE'] = False

    else:
        sent_info['tokens'].append(line.split())


def add_sgmMorphs(sent_info):
    tokens = copy.deepcopy(sent_info['tokens'])
    last = len(tokens) - 1
    segment_moprhs = []
    tokens.append(tokens[-1])  # dummy
    for idx, token in enumerate(tokens[:-1]):
        # print(token)
        try:
            morphs, pos, ne = token
        except:
            input(token)

        # if (morphs, pos) == ('_', '_'): #space
        #     morphs, pos = '</sp>', '</sp>'
        #
        # if (morphs != '</sp>') and ((tokens[idx + 1][0], tokens[idx + 1][1]) != ('_', '_')) and (idx != last):
        #     morphs += '</m>'
        #
        # segment_moprhs.append((morphs, pos, ne))


        if is_alphabet(morphs) or morphs.isdigit():
            morphs = list(morphs)
        if (morphs, pos) == ('_', '_'): #space
            morphs, pos = '</sp>', '</sp>'

        if type(morphs) == list:
            if ((tokens[idx+1][0], tokens[idx+1][1]) != ('_', '_')) and (idx != last):
                morphs[-1] = morphs[-1] + '</m>'
            for m in morphs:
                segment_moprhs.append((m, pos, ne))
        else:
            if (morphs != '</sp>') and ((tokens[idx+1][0], tokens[idx+1][1]) != ('_', '_')) and (idx != last):
                morphs += '</m>'
            segment_moprhs.append((morphs, pos, ne))
        # pprint(segment_moprhs)
        # input("-"*20)

    sent_info['sgmMorphs'] = segment_moprhs

    return sent_info


def readData(current_file):
    sent_info = {}
    sent_data = []

    # print(current_file)

    for line in open(current_file, mode='r', encoding='utf8').readlines():
        line = line.strip()
        if line == "":  ## end of data
            try:
                sent_data.append(add_sgmMorphs(sent_info))
            except IndexError as e:
                pass
            sent_info = {}
            continue

        decision_sent_info(line, sent_info)

    return sent_data

if __name__ == '__main__':
    from pprint import pprint
    setting_path = r'../../data/corpus/modu_NER/모두의_말뭉치_개체명_0001.txt'

    sentence_data = readData(setting_path)

    pprint(sentence_data)
