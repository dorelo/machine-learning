import pandas as pd
import numpy as np
from sklearn import decomposition, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV

'''
Read in data
'''
complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
complete_data = complete_data.drop(['confidence'], axis=1)

class_1 = complete_data.loc[complete_data['prediction'] == 1.00]
class_0 = complete_data.loc[complete_data['prediction'] == 0]

#print class_0

print "Imputing class 1"
class_1 = class_1.apply(lambda x: x.fillna(x.mean()))
class_1 = class_1.set_index('ID')

#print class_1

print "Imputing class 0"
class_0 = class_0.apply(lambda x: x.fillna(x.mean()))
class_0 = class_0.set_index('ID')

full_data = pd.DataFrame(class_0.append(class_1))

y = full_data['prediction']
X = full_data.drop(['prediction'], axis=1)




# print full_data

 pipe_gauss = Pipeline([
     ('scale', StandardScaler()),
     ('pca', decomposition.PCA(n_components=6, whiten=False)),
     ('clf', BaggingClassifier(GaussianNB()))
])

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', decomposition.PCA(n_components=6, whiten=False)),
    ('clf', CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=300, max_features=1.0, max_depth=6,
                                                              learning_rate=0.05, min_samples_leaf=150)))
])
pipe_ccv = Pipeline([
     ('scale', StandardScaler()),
     ('clf', CalibratedClassifierCV(BaggingClassifier(GradientBoostingClassifier(), n_jobs=-1, verbose=True),
                                    method='isotonic', cv=5))
 ])

gb_grid_params = {'clf__base_estimator__learning_rate': [0.1, 0.05, 0.02, 0.01],
             'clf__base_estimator__max_depth': [4, 6, 8],
             'clf__base_estimator__min_samples_leaf': [20, 50, 100, 150],
             'clf__base_estimator__max_features': [1.0, 0.3, 0.1]
             }

cv = KFold(n_splits=10, random_state=2)

grid = GridSearchCV(pipe,
                    param_grid=gb_grid_params,
                     cv=cv,
                     verbose=True,
                     n_jobs=40)

# grid.fit(X, y)
# print("Best score ", grid.best_score_)
# print("Best estimator ", grid.best_estimator_)
# print("Best params ", grid.best_params_)

# print pipe.get_params().keys()

'''
Read in testing data
'''
test_data = pd.read_csv('testing.csv')
test_data = test_data.set_index('ID')

print "Fitting"
pipe.fit(X, y)

'''
Make predictions
'''
print "Predicting"
predictions = pd.DataFrame({'prediction': pd.Series(pipe.predict(test_data))})

'''
Write to CSV
'''
print "Writing"
predictions.index += 1
predictions.to_csv('out.csv')


