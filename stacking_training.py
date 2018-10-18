import pandas as pd
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
Models
'''
mlp = Pipeline([
    ('scale', StandardScaler()),
    ('clf', MLPClassifier())
])

gradient_boost = Pipeline([
    ('scale', StandardScaler()),
    ('clf', GradientBoostingClassifier())
])

svm = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('model', svm.SVC(kernel='rbf', C=6.781303098802164, gamma=0.0002665116179542694))
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


print "Predicting"
predictions = pd.DataFrame({'prediction': pd.Series(
    cross_val_predict(mlp, X, y, cv=kfold, n_jobs=40, verbose=3))})
print "Writing"
predictions.index += 1
predictions.to_csv('mlp_cross_val.csv')

print "Predicting"
predictions = pd.DataFrame({'prediction': pd.Series(
    cross_val_predict(gradient_boost, X, y, cv=kfold, n_jobs=40, verbose=3))})
print "Writing"
predictions.index += 1
predictions.to_csv('gboost_cross_val.csv')

print "Predicting"
svm.fit(X, y)
predictions = pd.DataFrame({'prediction': pd.Series(
    cross_val_predict(svm, X, y, cv=kfold, n_jobs=40, verbose=3))})
print "Writing"
predictions.index += 1
predictions.to_csv('svm_cross_val.csv')

print "Predicting"
gnb.fit(X, y)
predictions = pd.DataFrame({'prediction': pd.Series(
    cross_val_predict(gnb, X, y, cv=kfold, n_jobs=40, verbose=3))})
print "Writing"
predictions.index += 1
predictions.to_csv('gnb_cross_val.csv')
