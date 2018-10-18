'''
Fit the LR to the training output
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Predictions for 4 clfs on training data
combined_data = pd.read_csv('combined_data.csv')
combined_data.set_index('ID')
#
learn_y = combined_data['prediction']
learn_X = combined_data.drop('prediction', axis=1)
#
clf = LogisticRegression()
print "Fitting"
clf.fit(learn_X, learn_y)

'''
Use the fitted LR to predict on the testing output
'''

print "Predicting"
produced_test_data = pd.read_csv('combined_test_out.csv')
# Predict against the new matrix of testing data output
predictions = pd.DataFrame(
    {'prediction': pd.Series(clf.predict(produced_test_data))})
print "Writing"
predictions.index += 1
predictions.to_csv('stacked_estimator.csv')
