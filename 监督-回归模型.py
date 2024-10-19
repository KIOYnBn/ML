from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from icecream import ic

boston = load_boston() # 该数据已经被移除了
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 数据标准化
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# LinearRegression model
Ir = LinearRegression()
Ir.fit(x_train, y_train)
Ir_y_predict = Ir.predict(x_test)

# SGDRegression model
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)

# LinearRegression model 评估
Ir.score = Ir.score(y_test, y_test)
r2_score = r2_score(y_test, Ir_y_predict)
mean_squared_error = mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(Ir_y_predict))
mean_absolute_error = mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(Ir_y_predict))

# SGDRegression model 评估
sgdr.score = sgdr.score(x_test, y_test)
# 同LR评估

# 向量机——线性核函数配置
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_predict = linear_svr.predict(x_test)

# 向量机——多项式核函数配置
poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train)
poly_svr_predict = poly_svr.predict(x_test)

# 向量机——径向基和函数配置
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_predict = rbf_svr.predict(x_test)

# linear_svr 评估
linear_svr.score(x_test, y_test)
mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(Ir_y_predict))
mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(Ir_y_predict))
# 其他同理

# k邻近算法
uni_knr = KNeighborsRegressor(weights='uniform') # 平均回归
# dis_knr.KNeighborsRegressor(weights='distance') 距离加权平均
uni_knr.fit(x_train, y_train)
uni_knr_y_predict = uni_knr.predict(x_test)

# k邻近算法评估
# 同上

# 回归树
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_predict = dtr.predict(x_test)

# 随机森林
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)

etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_predict = etr.predict(x_test)

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)

