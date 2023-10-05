import pandas as pd
from sklearn.linear_model import LogisticRegression

# 读取表1和表2中的数据
table1 = pd.read_excel('表1.xlsx') 
table2 = pd.read_excel('表2.xlsx')

# 将表1和表2进行合并
data = pd.merge(table1, table2, on='ID')

# 提取需要的特征
features = ['age', 'gender', 'history', ...] 

# 获得每个患者的首次影像时间和血肿体积
first_scan = data.groupby('ID')['time'].min()
first_volume = data[data['time'] == first_scan]['HM_volume']  

# 标记是否发生血肿扩张
data['hemo_increase'] = 0
for idx, row in data.iterrows():
    if row['HM_volume'] - first_volume.loc[row['ID']] >= 6 or (row['HM_volume'] - first_volume.loc[row['ID']]) / first_volume.loc[row['ID']] >= 0.33:
        data.loc[idx, 'hemo_increase'] = 1

# 拆分训练集和测试集        
train = data[data['ID'] <= 100]
test = data[data['ID'] > 100]

# 训练模型
# 使用XGBoost算法训练
params = {
    'eta': 0.1,
    'max_depth': 3, 
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain)

# 预测
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

# 输出结果
submit = pd.DataFrame({'ID': test['ID'], 'prob': y_pred})
submit.to_csv('submit.csv', index=False)