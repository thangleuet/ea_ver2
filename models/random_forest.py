from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics

class RandomForest():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=123)

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def test(self):
        self.y_pred = self.model.predict(self.x_test)
        print(pd.crosstab(self.y_test, self.y_pred, rownames=['Actual'], colnames=['Predicted']))
        print(metrics.classification_report(self.y_test, self.y_pred))
        print("Accuracy: " + str(metrics.accuracy_score(self.y_test, self.y_pred)))
       
                
    