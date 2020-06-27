import numpy as np
import math

# 决策树桩
class DecisionStump():
    def __init__(self):
        #基于划分阈值决定样本分类是1还是-1
        self.polarity = 1
        #特征索引
        self.feature_index = None
        #特征划分阈值
        self.threshold = None
        #指示分类准确率的值
        self.alpha = None

class AdaBoost():
    # 弱分类器个数
    def __init__(self,n_estimators=5):
        self.n_estimators = n_estimators

    # 拟合算法
    def fit(self,X,y):
        n_samples,n_features = X.shape

        # (1) 初始化权重分布为均匀分布 1/N
        w = np.full(n_samples,(1/n_samples))

        self.estimators = []

        # (2) for m in (1,2,,,m)
        for _ in range(self.n_estimators):
            # (2.a) 训练一个弱分类器，决策树桩
            clf = DecisionStump()
            # 设定一个最小化误差
            min_error = float('inf')
            # 遍历数据集特征，根据最小分类误差率选择最优划分特征
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:,feature_i],axis=1)
                unique_values = np.unique(feature_values)

                # 尝试将每一个特征值作为分类阈值
                for threshold in unique_values:
                    p = 1
                    # 初始化所有预测值为1
                    prediction =np.ones(np.shape(y))
                    # 小于分类阈值的预测值为1
                    prediction[X[:,feature_i]<threshold] = -1
                    # (2.b)计算误差率
                    error = sum(w[y!=prediction])
                    # 如果分类不差大于0。5，则进行正负预测反转
                    if error > 0.5:
                        error = 1- error
                        p = -1
                    # 一旦获得最小误差则保存相关参数设置
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error

            # (2.3) 计算基分类器的权重
            print("min_error:",min_error)
            clf.alpha = 0.5 * math.log((1.0-min_error)/(min_error+1e-10)) #避免分母为0
            # 初始化所有预测值为1,
            predictions = np.ones(np.shape(y))
            # 获取所有小于阈值的负类索引
            negative_idx = (clf.polarity*X[:,clf.feature_index]<clf.polarity*clf.threshold)
            # 将负类设置为 -1
            predictions[negative_idx] = -1
            #(2.d) 更新样本权重
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # 保存该弱分类器
            self.estimators.append(clf)

    # 定义预测函数
    def predict(self,X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples,1))
        # 计算每个弱分类器的预测值
        for clf in self.estimators:
            predictions = np.ones(np.shape(y_pred))
            # 获取所有小于阈值的负类索引
            negative_idx = (clf.polarity*X[:,clf.feature_index]<clf.threshold)
            # 将负类设置为 -1
            predictions[negative_idx] = -1
            #（2.e) 对每个弱分类器的预测结果进行甲醛
            y_pred += clf.alpha *predictions
        # 返回最终结果
        y_pred = np.sign(y_pred).flatten()
        return y_pred











