import pandas as pd
from sklearn import svm, model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import StackingClassifier

complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
print complete_data.shape
print complete_data


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

clf0 = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('clf', KNeighborsClassifier(n_neighbors=5))
    ])

clf1 = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=10, whiten=False)),
    ('clf', GaussianNB())
])

clf2 = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('clf', GradientBoostingClassifier())
])

clf3 = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('model', svm.SVC(kernel='rbf', C=6.781303098802164,
                      gamma=0.0002665116179542694, cache_size=10000))
])

clf4 = Pipeline([
    ('pca', PCA(n_components=6, whiten=False)),
    ('rf', RandomForestClassifier(random_state=1, n_estimators=50))])

lr = LogisticRegression(C=0.1)
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

print "Cross validating:\n"

for clf, label in zip([clf0, clf1, clf2, clf3, clf4, sclf],
                      ['KNN',
                       'GNB',
                       'Boost',
                       'SVM',
                       'RandomForest',
                       'StackingClassifier']):

    scores = cross_val_score(
        clf, X, y, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

print "Predicting"
sclf.fit(X, y)
predictions = pd.DataFrame({'prediction': pd.Series(sclf.predict(test_data))})
print "Writing"
predictions.index += 1
predictions.to_csv('out.csv')

params = {'randomforestclassifier__n_estimators': [10, 50],
          'meta-logisticregression__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf,
                     param_grid=params,
                     cv=5,
                     refit=True,
                     n_jobs=-1,
                     verbose=3)
 grid.fit(X, y)

  cv_keys = ('mean_test_score', 'std_test_score', 'params')

   for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)
