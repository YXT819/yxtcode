import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 固定随机种子以保证复现性
np.random.seed(0)

def toy_data():
    X1,y1=datasets.make_circles(n_samples=100, factor=.1, noise=.1)
    return X1, y1

def random_data():
    X2,y2=datasets.make_classification(n_features=2, n_redundant=0, n_informative=2,
                                          n_clusters_per_class=1, n_samples=100)
    return X2, y2

def iris_data():
    iris=datasets.load_iris()
    X3=iris.data[:, :2]  # 只取前两个特征简化问题
    y3=iris.target
    return X3, y3

def run_svm(X, y, kernel='linear', C=1.0):
    clf=SVC(kernel=kernel, C=C, random_state=0)
    clf.fit(X, y)
    return clf

# 参数列表
kernels=['linear', 'poly', 'rbf', 'sigmoid']
Cs = [0.1, 1, 10]

# 数据集
datasets_functions=[toy_data, random_data, iris_data]
dataset_names=['Toy data', 'Random data', 'Iris data']

# 运行实验
for data_func, name in zip(datasets_functions, dataset_names):
    X,y=data_func()
    print(f"\nDataset: {name}")
    for kernel in kernels:
        for C in Cs:
            clf=run_svm(X, y, kernel=kernel, C=C)
            print(f"Kernel: {kernel}, C: {C}, Support Vectors: {clf.support_vectors_.shape[0]}")

X_iris,y_iris=iris_data()
X_train,X_test,y_train,y_test=train_test_split(X_iris, y_iris, test_size=0.1, random_state=42)

best_score=0
best_params={}

for kernel in kernels:
    for C in Cs:
        clf=run_svm(X_train, y_train, kernel=kernel, C=C)
        score=clf.score(X_test, y_test)
        if score > best_score:
            best_score=score
            best_params={'kernel': kernel, 'C': C}

print(f"Best params: {best_params}, Best score: {best_score}")

X_rand,y_rand=random_data()
clf=run_svm(X_rand, y_rand, kernel=best_params['kernel'], C=best_params['C'])

plt.figure(figsize=(10, 6))
plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_rand, s=30, cmap=plt.cm.Paired)

# 描绘决策边界
ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

# 生成网格
xx,yy=np.meshgrid(np.linspace(xlim[0], xlim[1], 30),
                     np.linspace(ylim[0], ylim[1], 30))

# 预测网格点的函数值
Z=clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z=Z.reshape(xx.shape)

# 绘制决策边界和margin
ax.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'], colors='k')

# 高亮显示支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()