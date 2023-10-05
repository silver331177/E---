# 导入需要的库
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data.csv')

# 特征工程
X = data[['age', 'gender', 'treatment', 'image_features']]
y = data['mRS']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

# Lasso回归
model = Lasso()

# 使用网格搜索找到最优参数
from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.001, 0.01, 0.1, 1]}
gs = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
gs.fit(X_train, y_train)
print('最优参数:', gs.best_params_)
model = gs.best_estimator_

# 训练最优模型
model.fit(X_train, y_train)

# 交叉验证
scores = cross_val_score(model, X_train, y_train, cv=5)
print('交叉验证得分:', scores)
print('平均得分:', scores.mean())

# 学习曲线
from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(model, X_train, y_train, cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label='训练集得分')
plt.plot(train_sizes, valid_scores.mean(axis=1), label='验证集得分')
plt.legend()
plt.xlabel('训练集大小'); plt.ylabel('得分')
plt.title('Lasso回归学习曲线')
plt.show()

# 测试集评估
y_pred = model.predict(X_test)
print('测试集RMSE:', mean_squared_error(y_test, y_pred, squared=False))

# 特征重要性
importance = model.coef_ 
# 输出系数图
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), X.columns, rotation=90)
plt.show()

# 特征选择
model.fit(X_train, y_train) 
model = Lasso(alpha=0.01)
model.fit(X_train, y_train)
print('重要特征:', [f for f,v in zip(X.columns, model.coef_) if v != 0])

# 删除无关特征,比较模型效果变化
# ...