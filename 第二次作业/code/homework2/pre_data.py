import os
import re
import jieba
from collections import Counter
novel_labels = {
    '白马啸西风': 1,
    '碧血剑': 2,
    '飞狐外传': 3,
    '连城诀': 4,
    '鹿鼎记': 5,
    '三十三剑客图': 6,
    '射雕英雄传': 7,
    '神雕侠侣': 8,
    '书剑恩仇录': 9,
    '天龙八部': 10,
    '侠客行': 11,
    '笑傲江湖': 12,
    '雪山飞狐': 13,
    '倚天屠龙记': 14,
    '鸳鸯刀': 15,
    '越女剑': 16
}
def get_corpus_jieba(rootDir, total_segments=1000):
    corpus = []
    labels = []
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 过滤字符

    listdir = os.listdir(rootDir)
    num_files = len([f for f in listdir if f.endswith('.txt')])
    segments_per_file = max(1, total_segments // num_files + num_files)  # Ensure at least one segment per file

    for filename in listdir:
        if filename.endswith('.txt'):
            path = os.path.join(rootDir, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding='gbk', errors='ignore') as file:
                    filecontext = file.read()
                    filecontext = re.sub(r1, '', filecontext)
                    filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    tokens = list(jieba.cut(filecontext))
                    segment = []
                    count = 0
                    for token in tokens:
                        segment.append(token)
                        if len(segment) >= 1000:
                            corpus.append(segment)
                            labels.append(novel_labels[filename.replace('.txt', '')])  # Use the file name as the label
                            segment = []
                            count += 1
                            if count >= segments_per_file:
                                break  # Stop after adding required number of segments

    return corpus, labels
