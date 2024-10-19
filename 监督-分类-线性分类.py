import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from icecream import ic
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.datasets  import load_digits


# 数据预处理
column_names = [
    'Sample code number',
    'Clump Thickness',
    'Uniformity of Cell Size',
    'Uniformity of Cell Shape',
    'Marginal Adhesion',
    'Single Epithelial Cell Size',
    'Bare Nuclei',
    'Bland Chromatin',
    'Normal Nucleoli',
    'Mitoses',
    'Class'
]

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
ic(data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    data[column_names[1:10]],
    data[column_names[10]],
    test_size=0.25, random_state=33)

ic(y_train.value_counts(), y_test.value_counts())

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 逻辑斯蒂回归
Ir = LogisticRegression()
Ir.fit(x_train, y_train)
Ir_y_predict = Ir.predict(x_test)

# 随机梯度参数估计
sgdc = SGDClassifier()
sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)

# 线性分类模型性能分析-逻辑斯蒂回归评估
Ir.score = Ir.score(x_test, y_test) # 模型自带的评分函数score
classification_report = classification_report(y_test, Ir_y_predict, target_names=['benign', 'Malignant'])

# 线性分类模型性能分析-随机梯度参数估计评估
# 同上
