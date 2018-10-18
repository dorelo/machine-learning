# '''
# Concat data sets
# '''
# full_data_X = pd.concat([X, additional_data_X], axis=1)
# print "Concatenated X"
# #print full_data_X
#
# full_data_y = pd.concat([y, additional_data_y], axis=0)
# print "Concatenated y"
# #print full_data_y
#
# print("Full X shape: ", full_data_X.shape)
# print("Full y shape: ", full_data_y.shape)
#

# '''
# Train the model
# '''
# scaler = StandardScaler()
# full_data_X = scaler.fit_transform(full_data_X)
# print("Full X scaled shape: ", full_data_X.shape)

# #
# seed = 7
# kfold = KFold(n_splits=5, random_state=seed)
# results = cross_val_score(pipe, full_data_X, full_data_y, cv=kfold, n_jobs=-1, verbose=True)
# print results

# '''
# Training
# '''
# X_train, X_test, y_train, y_test = train_test_split(full_data_X, full_data_y, test_size=0.5, random_state=42)
#
#


#
#
# '''
# Make predictions
# '''
# predictions = pd.DataFrame({'prediction': pd.Series(pipe.predict(test_data))})
#
#
# '''
# Write to CSV
# '''
# predictions.index += 1
# predictions.to_csv('out.csv')

# '''
# CV SVM
# '''
# pipe = Pipeline([
#     ('scale', StandardScaler()),
#     ('pca', decomposition.RandomizedPCA()),
#     ('svm', svm.SVC()),
# ])

# C_range = np.logspace(-2, 10, 13)
# gamma_range = np.logspace(-9, 3, 13)
# cache_size = np.array(['svm__cache_size=7000'])
# param_grid = dict(svm__gamma=gamma_range, svm__C=C_range, svm__cache_size=cache_size)
# print pipe.get_params().keys()
# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# print "Grid searching..."
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=40, verbose=True)
# print "Fitting to data"
# grid.fit(full_data_X, full_data_y)
# print("Best score ", grid.best_score_)
# print("Best estimator ", grid.best_estimator_)
# print("Best params ", grid.best_params_)

# '''
# Train
# '''
# print "Fitting"
# pipe.fit(full_data_X, full_data_y)

# X_train, X_test, y_train, y_test = train_test_split(full_data_X, full_data_y)
# seed = 7
# kfold = KFold(n_splits=5, random_state=seed)
# results = cross_val_score(pipe, X_train, y_train, cv=kfold, n_jobs=40)
# print results

# '''Univariate Feature selection'''
# test = SelectKBest(score_func=chi2, k=1000)
# fit = test.fit(full_data_X, full_data_y)
# # summarize scores
# np.set_printoptions(precision=3)


seed = 7
pca_range = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 1000])
estimators = np.array([10, 20, 30, 50, 100])
whiten = np.array(['True', 'False'])
features = np.array([0.5, 0.6, 0.75])
samples = np.array([0.5, 0.6, 0.75])
param_grid = dict(pca__n_components=pca_range, bagging__n_estimators=estimators, pca__whiten=whiten,
                  bagging__max_features=features, bagging__max_samples=samples)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(pipe, param_grid=param_grid,
                    cv=cv, n_jobs=-1, verbose=True)
print "Fitting to data"
grid.fit(full_data_X, full_data_y)
print("Best score ", grid.best_score_)
print("Best estimator ", grid.best_estimator_)
print("Best params ", grid.best_params_)

np.logspace(np.log10(0.01), np.log10(1.0), num=20)

clf1 = GaussianNB()
clf2 = svm.SVC(kernel='rbf', C=1, gamma=0.0001, cache_size=7000)
clf3 = KNeighborsClassifier()

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', decomposition.PCA(n_components=6, whiten=False)),
    ('vote', VotingClassifier(estimators=[
        ('GNB', clf1),
        ('svm', clf2),
        ('knn', clf3)
    ])),
])
