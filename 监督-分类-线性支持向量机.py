from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from icecream import ic
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits = load_digits()
ic(digits.data.shape)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=33, test_size=0.25)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

ic(lsvc.score(x_test, y_test))
ic(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))


