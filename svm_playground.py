'''
Read in data
'''
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
print complete_data.shape
# print complete_data


complete_data = complete_data.loc[complete_data['confidence'] == 1.00]

# predict_1 = complete_data.loc[complete_data['prediction'] == 1]
# predict_0 = complete_data.loc[complete_data['prediction'] == 0]

'''
Impute missing values
'''
print "Imputing... 1"
complete_data = complete_data.apply(lambda x: x.fillna(x.mean()))
complete_data = complete_data.set_index('ID')
print "Impute complete"
#
# print "Imputing... 0"
# predict_0 = predict_0.apply(lambda x: x.fillna(x.mean()))
# predict_0 = predict_0.set_index('ID')
# print "Impute complete"

# print "Concatenating the two sets"
# final_data = pd.DataFrame(predict_1).append(predict_0)
print "Done"
# print final_data


y = complete_data['prediction']
X = complete_data.drop(['prediction', 'confidence'], axis=1)
#
# # print("Confidence shape", confidence.shape)
# # print("Data shape", complete_data.shape)
#
'''
Read in testing data
'''
test_data = pd.read_csv('testing.csv')
test_data = test_data.set_index('ID')
#
#

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('model', BaggingClassifier(svm.SVC(kernel='rbf',
                                        C=6.459355352710979, gamma=0.00043253636108190913)))
])

# seed = 2
# kfold = StratifiedKFold(n_splits=10, random_state=seed)
# print "Scoring"
# results = cross_val_score(pipe, X, y, cv=kfold, verbose=True, n_jobs=-1)
# print results
# print("Mean: ", np.mean(results))

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
