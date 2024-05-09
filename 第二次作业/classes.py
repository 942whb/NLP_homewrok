import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 假设 get_corpus_jieba 和 get_LDA_features 已经定义并且可以导入
from pre_data import get_corpus_jieba
from LDA import get_LDA_features
# 获取数据和特征
rootDir = '/home/whb/NLP_homework/homework2/data'
corpus, labels = get_corpus_jieba(rootDir,total_segments=1000)
features = get_LDA_features(corpus, num_topics=128, passes=10)
# 转换labels为numpy数组以便与scikit-learn兼
labels = np.array(labels)
# 创建支持向量机分类器实例
svm_classifier = SVC(kernel='linear', probability=True)  # 使用线性核
# 进行交叉验证
# cv参数决定了交叉验证分割的数量，这里使用10次
scores = cross_val_score(svm_classifier, features, labels, cv=10)
# 输出模型性能
print("平均准确率: %0.2f (标准差: %0.2f)" % (scores.mean(), scores.std()))