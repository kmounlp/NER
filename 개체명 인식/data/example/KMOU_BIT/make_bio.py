from pathlib import Path

def tags(name):
    return '{}.tags.txt'.format(name)
count = 0
print('Build vocab tags (may take a while)')
vocab_tags = set()
for n in ['train', 'testa', 'testb']:
    with Path(tags(n)).open(encoding="utf8") as f, Path(tags(n+"1")).open('w',encoding="utf8") as wf:
        for line in f:
            temp = []
            line = line.strip()
            count += len(line.split())
            for ele in line.split():
                if ele[1] == "-":
                    temp.append(ele)
                else:
                    temp.append("O")
            print(" ".join(temp),file=wf)
            vocab_tags.update(line.strip().split())
    print(count)
    count = 0
