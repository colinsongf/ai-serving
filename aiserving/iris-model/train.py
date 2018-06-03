from sklearn import datasets
from sklearn import svm
from aiserving import model_io

iris = datasets.load_iris()
print(iris.DESCR)
clf = svm.SVC(gamma=0.001, C=100.,probability=True)
clf.fit(iris.data[:-1], iris.target[:-1])
model_io.save_pickle_model(clf, "iris_model.pkl", "./resources/pickle")
print("saved model success")
