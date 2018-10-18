import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
from glob import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


complete_data = pd.read_csv('full_data_with_confidence.csv')
complete_data.set_index('ID')
print complete_data.shape


'''
Impute missing values
'''
print "Imputing... 1"
complete_data = complete_data.apply(lambda x: x.fillna(x.mean()))
complete_data = complete_data.set_index('ID')
print "Impute complete"

complete_data = complete_data.loc[complete_data['confidence'] == 1.00]

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
    ('clf', GradientBoostingClassifier(n_estimators=100))
])

svm_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=6, whiten=False)),
    ('model', svm.SVC(kernel='rbf', C=6.781303098802164,
                      gamma=0.0002665116179542694, probability=True))
])

gnb = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=10, whiten=False)),
    ('clf', BaggingClassifier(GaussianNB()))
])

'''
Cross validation
'''


def cross_validate(model):
    seed = 2
    kfold = StratifiedKFold(n_splits=10, random_state=seed)
    print "Scoring"
    results = cross_val_score(model, X, y, cv=kfold, verbose=True, n_jobs=-1)
    print results
    print("Mean: ", np.mean(results))


'''
Write predictions to output file
'''


def output(*params):
    counter = 0
    for param in params:
        print "Predicting.."
        param.fit(X, y)
        predictions = pd.DataFrame(
            {'prediction': pd.Series(param.predict(test_data))})
        print "Writing"
        predictions.index += 1
        predictions.to_csv('method_' + repr(counter) + '.csv')
        print "Successfully wrote"
        counter = counter + 1


'''
Average the scores "vote"
'''


def vote(input_files, outfile, method="average"):
    pattern = re.compile(r"(.)*_[w|W](\d*)_[.]*")
    if method == "average":
        scores = defaultdict(list)
    with open(outfile, "wb") as outfile:
        weight_list = [1] * len(glob(input_files))
        for i, glob_file in enumerate(glob(input_files)):
            print "parsing:", glob_file
            lines = open(glob_file).readlines()
            lines = [lines[0]] + sorted(lines[1:])
            for e, line in enumerate(lines):
                if i == 0 and e == 0:
                    outfile.write(line)
                if e > 0:
                    row = line.strip().split(",")
                    for l in range(1, weight_list[i] + 1):
                        scores[(e, row[0])].append(row[1])
        for j, k in sorted(scores):
            outfile.write("%s,%s\n" %
                          (k, Counter(scores[(j, k)]).most_common(1)[0][0]))
        print("wrote to %s" % outfile)


'''
ROC curve
'''


def roc_gen(X_val, y_val, classifier):
    X_val = X_val.as_matrix()
    y_val = y_val.as_matrix()
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = classifier

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen',
                    'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0
    for (train, test), color in zip(cv.split(X_val, y_val), colors):
        probas_ = classifier.fit(
            X_val[train], y_val[train]).predict_proba(X_val[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_val[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(X_val, y_val)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic SVM')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_unoptimized.png')


'''
Determine correlations
'''


def correlation(first_file, second_file):
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    prediction = first_df.columns[0]
    print "Finding correlation between: %s and %s" % (first_file, second_file)
    print "Column to be measured: %s" % prediction
    print "Pearson's correlation score: %0.5f" % first_df[prediction].corr(second_df[prediction], method='pearson')


'''
Call methods
'''
#roc_gen(X, y, svm_pipe_unoptimized)

input_files = "./method_*.csv"
outfile = "./vote_output.csv"

output(gradient_boost, svm_pipe, gnb)
vote(input_files, outfile)

# compute_roc()

# cross_validate(gnb)

# output(svm_pipe, gradient_boost)
# file1 = "./method_0.csv"
# file2 = "./method_1.csv"
# correlation(file1, file2)
quit()
