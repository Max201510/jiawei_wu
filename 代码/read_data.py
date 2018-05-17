from sklearn import preprocessing, cross_validation, svm, metrics
from sklearn.linear_model import LinearRegression
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
data = pd.read_csv('learn_motivation.csv')
print(corrcoef([data.score,data.self_test,data.motivation]))

def draw_cof(data):
    plt.scatter(data.score, data.motivation)
    plt.plot([75, 88], [60, 92], c='r')
    plt.plot([62, 77], [77, 107], c='r')
    plt.xlabel('学分加权平均分', fontsize=18)
    plt.ylabel('职业规划调查得分', fontsize=18)
    plt.show()
    plt.scatter(data.self_test, data.motivation, )
    plt.xlabel('学习动机自我诊断', fontsize=18)
    plt.ylabel('职业规划调查得分', fontsize=18)
    plt.plot([-1.2, 8.44], [90.52, 63.62], c='r')
    plt.plot([8.84, 15.1], [104, 81.7], c='r')
    plt.show()

# draw_cof(data)


X = np.array(data[['score','self_test']])
y = np.array(data['motivation'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression()
clf.fit(X_train, y_train)
a, b = clf.coef_
c = clf.intercept_
print(a,b,c)
y_pre = clf.predict(X_test)
mean_sprt_error = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
mean_true = y_test.mean()
print(mean_sprt_error/mean_true)

