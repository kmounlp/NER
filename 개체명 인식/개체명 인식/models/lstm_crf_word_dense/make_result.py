with open('label_test.txt','r') as rf, open('results.txt', 'w') as wf:
    for line in rf.readlines():
        line = line.strip()
        word, tag, pred = line.split()
        if len(tag) == 1:
            tag = 'O'
        else:
            if tag[1] != "-":
                tag = "O"
        if len(pred) == 1:
            pred = 'O'
        else:
            if pred[1] != "-":
                pred = "O"
        print(" ".join([word, tag, pred]),file=wf)
        
        
        
