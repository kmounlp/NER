train_words = set()
test_words = set()
test_words_list = []
line_lists = []
train_all = []
test_all = []
train_tag = []
test_tag = []
with open("train.words.txt",'r',encoding="utf8") as f:
    for line in f.readlines():
        train_all.extend(line.strip().split())
with open("train.tags.txt",'r',encoding="utf8") as f:
    for line in f.readlines():
        train_tag.extend(line.strip().split())  
with open("testb.words.txt",'r',encoding="utf8") as f:
    for line in f.readlines():
        test_all.extend(line.strip().split())
with open("testb.tags.txt",'r',encoding="utf8") as f:
    for line in f.readlines():
        test_tag.extend(line.strip().split())
line_lists = []
for w, t in zip(train_all, train_tag):
    if t != "O":
        line_lists.append(w)
train_words.update(line_lists)
line_lists = []
for w, t in zip(test_all, test_tag):

    if t != "O":
        line_lists.append(w)
test_words.update(line_lists)
print(len(train_words), len(test_words))
diff = test_words.difference(train_words)
print(len(test_words),len(diff))
with open("kmou_oov", 'w',encoding="utf8") as f:
    for ele in diff:
        print(ele,file=f)
