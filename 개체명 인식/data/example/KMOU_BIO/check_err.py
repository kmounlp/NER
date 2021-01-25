from pathlib import Path
def words(name):
    return '{}.words.txt'.format(name)
def poss(name):
    return '{}.pos.txt'.format(name)
for n in ['train', 'testa', 'testb']:
    print(n)
    with Path(words(n)).open(encoding="utf8") as f, Path(poss(n)).open(encoding="utf8") as pf:
        for line,pline in zip(f,pf):
##            print(line,pline)
            if len(line.strip().split()) != len(pline.strip().split()):
                print(len(line.strip().split()))
                print(len(pline.strip().split()))
                print(line,pline)
                print("__________________")
