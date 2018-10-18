import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    BaggingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, decomposition
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC, NuSVC
from sklearn.preprocessing import Imputer
import timeit
import httplib
import urllib
from sklearn.metrics import log_loss

'''
Read in training data
'''

'''
Read in data
'''
complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
print complete_data.shape
# print complete_data
'''
Impute missing values
'''
print "Imputing..."
complete_data = complete_data.apply(lambda x: x.fillna(x.mean()))
complete_data = complete_data.set_index('ID')
print "Impute complete"

#confidence = complete_data['confidence']

#full_confidence_data = complete_data.loc[complete_data['confidence'] == 1.00]
# print full_confidence_data

#duplicated_confidence_data = complete_data.append(full_confidence_data, ignore_index=True)
# Reset the index value to start at 1
#duplicated_confidence_data.index = np.arange(1, len(duplicated_confidence_data) + 1)

#complete_data = complete_data.drop(complete_data.loc[complete_data['confidence'] != 1.00], axis=0)

# print "Dropped"
#
y = complete_data['prediction']
X = complete_data.drop(['prediction', 'confidence'], axis=1)

# print("Confidence shape", confidence.shape)
# print("Data shape", complete_data.shape)

'''
Read in testing data
'''
test_data = pd.read_csv('testing.csv')
test_data = test_data.set_index('ID')

'''
CV
'''
clf1 = GaussianNB()
clf2 = GradientBoostingClassifier()
clf3 = LogisticRegression()
# clf2 = svm.SVC(kernel='rbf', C=1, gamma=0.0001, cache_size=7000, verbose=True)
# clf3 = LogisticRegression()
# ('vote', VotingClassifier(estimators=[('gnb', clf1), ('gbc', clf2), ('lr', clf3)],
#                               voting='soft',
#                               weights=[1,2,1]))])

# pipe = Pipeline([
#     ('scale', StandardScaler()),
#     ('pca', decomposition.PCA(n_components=6, whiten=False)),
#     ('svc', BaggingClassifier(NuSVC(gamma=0.0001, cache_size=7000)))])


# clf1 = GaussianNB()
# clf2 = svm.SVC(kernel='rbf', C=1, gamma=0.0001, cache_size=7000)
# clf3 = KNeighborsClassifier()
#
# pipe = Pipeline([
#     ('scale', StandardScaler()),
#     ('pca', decomposition.PCA(n_components=6, whiten=False)),
#     ('vote', VotingClassifier(estimators=[
#         ('GNB', clf1),
#         ('svm', clf2),
#         ('knn', clf3)
#     ])),
# ])

# GNB
gnb = Pipeline([
    ('scale', StandardScaler()),
    ('pca', decomposition.PCA(n_components=10, whiten=False)),
    ('clf', BaggingClassifier(GaussianNB()))
])

# Split Train / Test
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=36)


gradient_boost = Pipeline([
    ('scale', StandardScaler()),
    ('clf', CalibratedClassifierCV(BaggingClassifier(
        GradientBoostingClassifier(), n_jobs=40), method='isotonic', cv=5))
])

# gradient_boost.fit(Xtrain, ytrain)
# ypreds = gradient_boost.predict_proba(Xtest)
# print("loss WITHOUT calibration : ", log_loss(ytest, ypreds, eps=1e-15, normalize=True))

# X_new = SelectKBest(chi2, k=20).fit_transform(X, y)


seed = 2
kfold = KFold(n_splits=10, random_state=seed)
print "Scoring"
results = cross_val_score(pipe, X, y, cv=kfold, verbose=True, n_jobs=-1)
print results
print("Mean: ", np.mean(results))


#('svm', svm.SVC(kernel='rbf', C=1, gamma=0.0001, cache_size=7000, verbose=True))

# #
# print "Fitting"
# gnb.fit(X, y)
# #
# # #print pipe.get_params().keys()
# #
# #
# #
# #
#
# '''
# Make predictions
# '''
# print "Predicting"
# predictions = pd.DataFrame({'prediction': pd.Series(gnb.predict(test_data))})
#
#
# '''
# Write to CSV
# '''
# print "Writing"
# predictions.index += 1
# predictions.to_csv('out.csv')
