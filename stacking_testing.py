import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
print complete_data.shape
# print complete_data


complete_data = complete_data.loc[complete_data['confidence'] == 1.00]

'''
Impute missing values
'''
print "Imputing... 1"
complete_data = complete_data.apply(lambda x: x.fillna(x.mean()))
complete_data = complete_data.set_index('ID')
print "Impute complete"

y = complete_data['prediction']
X = complete_data.drop(['prediction', 'confidence'], axis=1)

'''
Read in testing data
'''
test_data = pd.read_csv('testing.csv')
test_data = test_data.set_index('ID')


'''
Models
'''

gradient_boost = Pipeline([
    ('scale', StandardScaler()),
    ('clf', GradientBoostingClassifier())
])

svm = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('model', svm.SVC(kernel='rbf', C=6.459355352710979, gamma=0.00043253636108190913))
])

gnb = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=10, whiten=False)),
    ('clf', BaggingClassifier(GaussianNB()))
])


'''
Cross val
'''
seed = 2
kfold = StratifiedKFold(n_splits=10, random_state=seed)


'''
Predictions
'''
print "Predicting"
mlp.fit(X, y)
predictions = pd.DataFrame({'prediction': pd.Series(mlp.predict(test_data))})
print "Writing"
predictions.index += 1
predictions.to_csv('mlp_preds.csv')

print "Predicting"
gradient_boost.fit(X, y)
predictions = pd.DataFrame(
    {'prediction': pd.Series(gradient_boost.predict(test_data))})
print "Writing"
predictions.index += 1
predictions.to_csv('gboost_preds.csv')

print "Predicting"
svm.fit(X, y)
predictions = pd.DataFrame({'prediction': pd.Series(svm.predict(test_data))})
print "Writing"
predictions.index += 1
predictions.to_csv('svm_preds.csv')

print "Predicting"
gnb.fit(X, y)
predictions = pd.DataFrame({'prediction': pd.Series(gnb.predict(test_data))})
print "Writing"
predictions.index += 1
predictions.to_csv('gnb_preds.csv')
