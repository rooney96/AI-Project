import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

from constants import HYPER_PARAMETERS
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, plot_confusion_matrix, \
    f1_score


def select_best_k_features():
    knn = KNeighborsClassifier(n_neighbors=18)
    random_forest = RandomForestClassifier(criterion='gini', max_depth=7, min_samples_split=8, n_estimators=12)
    decision_tree = DecisionTreeClassifier(max_depth=6)
    svc = SVC(C=0.5, degree=1, gamma='scale', kernel='linear', probability=True)
    gb = GradientBoostingClassifier(max_depth=7, min_samples_leaf=0.006733, min_samples_split=0.0251)
    nb = GaussianNB()

    data = pd.read_excel(
        'C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\training_data.xlsx')
    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "Unnamed: 0", "FTHG", "FTAG"])
    # filling missing value
    x_train = x_train.fillna(x_train.mean())
    # normalization
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x_train.values)
    # x_train = pd.DataFrame(x_scaled)
    y_train = data["FTR"]
    features_len = len(x_train.columns)
    print(features_len)

    for k in range(15, features_len):
        best_features = SelectKBest(f_classif, k=k)
        fit = best_features.fit(x_train, y_train)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(x_train.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        selected_features = (featureScores.nlargest(k, 'Score'))['Specs']
        x_new = x_train[selected_features]

        results = []
        for clf in [knn, random_forest, decision_tree, svc, gb, nb]:
            clf.fit(x_new, y_train)
            scores = cross_val_score(clf, x_new, y_train, cv=5)
            results.append(scores.mean())

        print(f"Results for k={k}:")
        print(results)
        print(f"Top score is: {sorted(results)[-1]}")


def find_best_model():
    data = pd.read_excel('C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\training_data.xlsx')
    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "Unnamed: 0", "FTHG", "FTAG"])
    x_train = x_train.fillna(x_train.mean())
    y_train = data["FTR"]

    classifiers = {
        "KNN": KNeighborsClassifier,
        "DT": DecisionTreeClassifier,
        "RF": RandomForestClassifier,
        "SVC": SVC,
        "GB": GradientBoostingClassifier
    }
    scores = []
    for clf_name, clf in classifiers.items():
        param_finder = GridSearchCV(
            clf(),
            HYPER_PARAMETERS[clf_name],
            cv=StratifiedShuffleSplit(n_splits=5),
            return_train_score=True,
            verbose=2 ** 31,
            n_jobs=-1,
            refit=True
        )
        param_finder.fit(x_train, y_train)
        params, score = param_finder.best_params_, param_finder.best_score_
        chosen_clf = param_finder.best_estimator_

        clf_results = pd.concat([pd.DataFrame(param_finder.cv_results_["params"]),
                                 pd.DataFrame(param_finder.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
        clf_results = clf_results.sort_values(by='Accuracy', ascending=False)
        clf_results.to_excel(f"{clf_name}_results.xlsx")
        scores.append((chosen_clf, params, score))
    print(scores)
    return sorted(scores, key=lambda c: c[2], reverse=True)[0]


if __name__ == '__main__':
    # select_best_k_features()
    find_best_model()
    # data = pd.read_excel('C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\training_data.xlsx')
    # x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "Unnamed: 0", "FTHG", "FTAG"])
    # x_train = x_train.fillna(x_train.mean())
    # y_train = data["FTR"]

    """
    data = pd.read_excel('C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\training_data.xlsx')

    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "Unnamed: 0", "FTHG", "FTAG"] )
    x_train = x_train.reset_index(drop=True)
    y_train = data["FTR"]
    x_train = x_train.fillna(x_train.mean())

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(f"r-f accuracy: {scores}, {scores.mean()}")

    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    tree.plot_tree(clf)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(f"d-t accuracy: {scores}, {scores.mean()}")

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(f"KNN accuracy: {scores}, {scores.mean()}")

    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(f"GB accuracy: {scores}, {scores.mean()}")

    clf = SVC()
    clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(f"SVC accuracy: {scores}, {scores.mean()}")

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print(f"Naive Bayes accuracy: {scores}, {scores.mean()}")
    """
