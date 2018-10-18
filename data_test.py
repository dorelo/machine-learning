import pandas as pd
import numpy as np
from sklearn import decomposition, svm
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

'''
Read in data
'''
complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
print complete_data.shape
# print complete_data

'''
Remove low confidence
'''
# fixed = complete_data.loc[complete_data['confidence'] == 1.00]
# print fixed

'''
Impute missing values
    [*] This replaces empty values with the mean
'''
print "Imputing..."
complete_data = complete_data.apply(lambda x: x.fillna(x.mean()))
complete_data = complete_data.set_index('ID')
print "Impute complete"

# full_confidence_data = complete_data.loc[complete_data['confidence'] == 1.00]
# # print full_confidence_data
#
# duplicated_confidence_data = complete_data.append(full_confidence_data, ignore_index=True)
# # Reset the index value to start at 1
# duplicated_confidence_data.index = np.arange(1, len(duplicated_confidence_data) + 1)
# print "Appended"

# print duplicated_confidence_data

y = complete_data['prediction']
X = complete_data.drop(['prediction', 'confidence'], axis=1)

#nu = svm.NuSVC(kernel='rbf', gamma=0.0001)
gnb = GaussianNB(priors=[0.4286, 0.5714])

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', decomposition.PCA(n_components=6, whiten=False)),
    ('clf', BaggingClassifier(gnb))
])



seed = 2
kfold = KFold(n_splits=10, random_state=seed)
print "Scoring"
results = cross_val_score(pipe, X, y, cv=kfold, verbose=True, n_jobs=-1)
print results
print("Mean: ", np.mean(results))

# pipe.fit(X, y)
#
# '''
# Read in testing data
# '''
# test_data = pd.read_csv('testing.csv')
# test_data = test_data.set_index('ID')
#
# '''
# Make predictions
# '''
# print "Predicting"
# predictions = pd.DataFrame({'prediction': pd.Series(pipe.predict(test_data))})
#
#
# '''
# Write to CSV
# '''
# print "Writing"
# predictions.index += 1
# predictions.to_csv('out.csv')