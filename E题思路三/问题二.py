import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# 读取数据
data = pd.read_excel('table2.xlsx') 

# 特征工程:提取时间和水肿体积
X = data[['time']]  
y = data[['ED_volume']]

# 构建线性回归模型
lr = LinearRegression()

# 训练模型
lr.fit(X, y)

# 获取拟合的系数
print('模型Slope:', lr.coef_)  
print('模型Intercept:', lr.intercept_)

# 预测水肿体积
y_pred = lr.predict(X) 

# 可视化拟合效果
plt.scatter(X, y)  
plt.plot(X, y_pred, c='r')
plt.show()

# 计算残差
residual = y - y_pred  

# 绘制残差分布图
plt.hist(residual)
plt.show()

# K-Means进行分群
kmeans = KMeans(n_clusters=4)

# 训练,获得标签
clusters = kmeans.fit_predict(residual.values.reshape(-1,1))

# 添加分群结果到原始数据
data['cluster'] = clusters  

# 绘制分群结果
plt.scatter(X, y, c=clusters)
plt.show()

# 分析各群体大小
print(data.groupby('cluster').size())

# 每组单独建模
for i in range(4):
    Xi = X[data['cluster']==i]
    yi = y[data['cluster']==i] 
    model = LinearRegression()
    model.fit(Xi, yi)
    print('Cluster {}: y={:.2f}x+{:.2f}'.format(i, model.coef_[0], model.intercept_))
    
# 将治疗方法转为哑变量    
dummies = pd.get_dummies(data['treatment'])
data = pd.concat([data, dummies], axis=1)

# 建立回归模型分析治疗和水肿体积  
X2 = data[['ED_volume', 'treatment_A', 'treatment_B']]
lr2 = LinearRegression()
lr2.fit(X2, y)
print(lr2.coef_)