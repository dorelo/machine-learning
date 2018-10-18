from argparse import ArgumentError

import pandas as pd
import optunity
import optunity.metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler

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
fixed = complete_data.loc[complete_data['confidence'] == 1.00]
# print fixed
'''
Impute missing values
'''
print "Imputing..."
fixed = fixed.apply(lambda x: x.fillna(x.mean()))
fixed = fixed.set_index('ID')
print "Impute complete"


y = fixed['prediction']
X = pd.DataFrame(fixed.drop(['prediction', 'confidence'], axis=1))

y = y.as_matrix()
X = X.as_matrix()

scaler = StandardScaler()


data = X
data = scaler.fit_transform(data)
labels = y

space = {'kernel': {'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                    }
         }


cv_decorator = optunity.cross_validated(x=data, y=labels, num_folds=5)


def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = svm.SVC(kernel=kernel, C=C, cache_size=10000, verbose=3)
    elif kernel == 'poly':
        model = svm.SVC(kernel=kernel, C=C, degree=degree,
                        coef0=coef0, cache_size=10000, verbose=3)
    elif kernel == 'rbf':
        model = svm.SVC(kernel=kernel, C=C, gamma=10 **
                        logGamma, cache_size=10000, verbose=3)
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    model.fit(x_train, y_train)
    return model


def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='rbf', C=0, logGamma=0, degree=0, coef0=0):
    model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)


print "Cross validating"
svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

print "Determining optimal params"
optimal_svm_pars, info, _ = optunity.maximize_structured(
    svm_tuned_auroc, space, num_evals=150)
print("Optimal parameters" + str(optimal_svm_pars))
print("AUROC of tuned SVM: %1.3f" % info.optimum)

print "Logging"
df = optunity.call_log2dataframe(info.call_log)
print df
df.to_csv('gridsearch_results.csv')
