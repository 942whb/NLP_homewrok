from collections import Counter
import jieba

# 加载停用词列表
with open('/stop_word/cn_stopwords.txt', 'r', encoding='gbk') as f:
    stopwords = [line.strip() for line in f.readlines()]

with open('/data/inf.txt', 'r', encoding='gbk') as file:
    book_titles = [line.strip() for line in file.readlines()]

#用一个list来放所有的内容
words = []

for book_name in book_titles:
    with open('/data/%s.txt' %(book_name), 'r', encoding='gbk') as f:
        text = f.read()
    # 使用jieba进行分词
    words += jieba.lcut(text)

# 移除停用词并统计词频
filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
word_counts = Counter(filtered_words)

# 获取词频最高的30个词
top_30_words = word_counts.most_common(30)