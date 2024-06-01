import re
import jieba
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 假设我们已经有一个包含金庸小说文本的文件 "jin_yong_novels.txt"
with open('jin_yong_novels.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 文本清洗和分词
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 去除多余的空格
    words = jieba.lcut(text)  # 使用结巴分词
    return words

words = preprocess_text(text)

# 保存分词后的文本
with open('preprocessed_jin_yong.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(words))

# 训练Word2Vec模型
word2vec_model = Word2Vec([words], vector_size=100, window=5, min_count=5, workers=4)

# 保存模型
word2vec_model.save("word2vec_jin_yong.model")

# 加载模型
word2vec_model = Word2Vec.load("word2vec_jin_yong.model")

import gensim.downloader as api

# 下载预训练的GloVe模型
glove_vectors = api.load("glove-wiki-gigaword-100")

# 将金庸语料中的词映射到GloVe词向量
glove_word_vectors = {word: glove_vectors[word] for word in words if word in glove_vectors}

# 计算词向量之间的语义距离
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

word1 = "郭靖"
word2 = "黄蓉"
similarity = cosine_similarity(word2vec_model.wv[word1], word2vec_model.wv[word2])
print(f"'{word1}' 和 '{word2}' 的语义相似度: {similarity}")
from sklearn.cluster import KMeans

# 使用PCA降维到2维，以便可视化
def plot_words(model, words):
    word_vectors = np.array([model.wv[word] for word in words])
    pca = PCA(n_components=2)
    word_vectors_pca = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 10))
    plt.scatter(word_vectors_pca[:, 0], word_vectors_pca[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(word_vectors_pca[i, 0], word_vectors_pca[i, 1]))
    plt.show()

words_to_plot = ["郭靖", "黄蓉", "杨过", "小龙女", "乔峰", "虚竹", "段誉"]
plot_words(word2vec_model, words_to_plot)
def paragraph_vector(paragraph, model):
    words = [word for word in paragraph if word in model.wv]
    return np.mean([model.wv[word] for word in words], axis=0)

paragraph1 = "郭靖与黄蓉的故事"
paragraph2 = "杨过与小龙女的故事"
vec1 = paragraph_vector(jieba.lcut(paragraph1), word2vec_model)
vec2 = paragraph_vector(jieba.lcut(paragraph2), word2vec_model)

similarity = cosine_similarity(vec1, vec2)
print(f"段落 '{paragraph1}' 和 '{paragraph2}' 的语义相似度: {similarity}")

