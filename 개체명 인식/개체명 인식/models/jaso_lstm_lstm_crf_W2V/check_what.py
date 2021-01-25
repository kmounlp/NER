##a = list()
##with open("vocab.tags.txt",'r',encoding="utf8") as f:
##    for line in f.readlines():
##        line = line.strip()
##        
##        a.append(line)
##        
##a = set(a)
##print(len(a))

##with open("train.tags.txt",'r',encoding="utf8") as f:
##    for line in f.readlines():
##        line= line.strip()
##        for ele in line.split():
##            if ele in a:
##                continue
##            else:
##                print(line)

##with open("testa.tags.txt",'r',encoding="utf8") as f:
##    for line in f.readlines():
##        line= line.strip()
##        for ele in line.split():
##            if ele in a:
##                continue
##            else:
##                print(line)

#이건 한글
##with open("testb.preds.txt",'r',encoding="utf8") as rf, open("results.txt",'w',encoding="utf8") as wf:
##    for line in rf.readlines():
##        line= line.strip()
##        if len(line) == 0:
##            print(file=wf)
##            continue
##        word, answer, pred = tuple(line.split())
##        if word == "<SP>":
##            continue
##        if answer[1] != '_':
##            answer = 'O'
##        if pred[1] != '_':
##            pred = 'O'
##        print("{} {} {}".format(word, answer, pred),file=wf)

#이건 영어
with open("./results_tag_500_3/score/testb.preds.txt",'r',encoding="utf8") as rf, open("results.txt",'w',encoding="utf8") as wf:
    for line in rf.readlines():
        line= line.strip()
        if len(line) == 0:
            print(file=wf)
            continue
        word, answer, pred = tuple(line.split())
     
        if word == "<SP>":
            continue
        if len(answer) == 1:
            answer = 'O'
        else:
            if answer[1] != '_':
                answer = 'O'
        if len(pred) == 1:
            pred = 'O'
        else:
            if pred[1] != '_':
                pred = 'O'
        print("{} {} {}".format(word, answer, pred),file=wf)
