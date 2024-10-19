from icecream import ic
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

# 读取泰坦尼克号乘客档案
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
x['age'].fillna(x['age'].mean(), inplace=True) # 对于缺失年龄信息，使用平均年龄代替，尽量不影响预测人物。
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)

ic(dtc.score(x_test, y_test))
ic(classification_report(dtc_y_pred, y_test))

ic(rfc.score(x_test, y_test))
ic(classification_report(rfc_y_pred, y_test))

ic(gbc.score(x_test, y_test))
ic(classification_report(gbc_y_pred, y_test))