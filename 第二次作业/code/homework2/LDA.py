from pre_data import get_corpus_jieba
from gensim import corpora, models
import numpy as np
# 假设 corpus 是已经分好词的文档列表
# 创建字典
def get_LDA_features(corpus,num_topics = 10,passes=10):
    dictionary = corpora.Dictionary(corpus)
# 通过字典将文档转换为词袋模型
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
# 指定LDA模型的主题数 T
    num_topics = num_topics # 根据需求设置这个值
# 训练LDA模型
    lda = models.LdaModel(corpus_bow, num_topics=num_topics, id2word=dictionary, passes=passes)
# 为每个段落计算主题分布
    topic_distributions = [lda.get_document_topics(bow, minimum_probability=0.0) for bow in corpus_bow]
# 将主题分布转换为固定长度的向量
    topic_features = np.zeros((len(corpus), num_topics))
    for i, doc_topics in enumerate(topic_distributions):
        for topic in doc_topics:
            topic_id, prob = topic  # 解包元组
            topic_features[i, topic_id] = prob
    return topic_features