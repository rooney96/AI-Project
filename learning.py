import pandas as pd
from features_predictions import DataNormalizer
from constants import HYPER_PARAMETERS
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif


def select_best_k_features():
    knn = KNeighborsClassifier(n_neighbors=17, weights='distance')
    random_forest = RandomForestClassifier(criterion='gini', max_depth=7, min_samples_split=4, n_estimators=15)
    decision_tree = DecisionTreeClassifier(max_depth=5, splitter='random')
    svc = SVC(C=0.8, degree=1, gamma='auto', kernel='linear', probability=False)
    gb = GradientBoostingClassifier(max_depth=7, min_samples_leaf=0.01, min_samples_split=0.0002, n_estimators=100)
    ab = AdaBoostClassifier(n_estimators=30)
    data = pd.read_excel('learning_ai_algorithms_data\\training_data.xlsx')
    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "Unnamed: 0", "FTHG", "FTAG"])
    # filling missing value
    x_train = x_train.fillna(x_train.mean())
    # normalization
    data_normlizer = DataNormalizer()
    x_train = data_normlizer(x_train)
    y_train = data["FTR"]
    features_len = len(x_train.columns)
    print(features_len)

    pd_columns = ['K', 'KNN acc', 'RF acc', 'DT acc', 'SVC Acc', 'GB Acc', 'AB Acc', 'Best Acc']
    pd_data = []
    for k in range(10, 11):
        best_features = SelectKBest(f_classif, k=k)
        fit = best_features.fit(x_train, y_train)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(x_train.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        selected_features = (featureScores.nlargest(k, 'Score'))['Specs']
        print(selected_features)
        x_new = x_train[selected_features]

        results = []
        max_res = 0
        for clf in [knn, random_forest, decision_tree, svc, gb, ab]:
            clf.fit(x_new, y_train)
            scores = cross_val_score(clf, x_new, y_train, cv=10)
            results.append(scores.mean())
            if scores.mean() > max_res:
                max_res = scores.mean()
                max_k = k

        print(f"Results for k={k}:")
        print(results)
        best_acc = sorted(results)[-1]
        pd_data.append([k] + results + [best_acc])
        print(pd_data)
        print(f"Top score is: {best_acc}")

    best_features = SelectKBest(f_classif, k=max_k)
    fit = best_features.fit(x_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_train.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    selected_features = (featureScores.nlargest(max_k, 'Score'))['Specs']
    df = pd.DataFrame(data=pd_data, columns=pd_columns)
    df = df.sort_values(by='Best Acc', ascending=False)
    df.to_excel('SelectKBestResults.xlsx')


def find_best_model():
    data = pd.read_excel('learning_ai_algorithms_data\\training_data.xlsx')
    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "Unnamed: 0", "FTHG", "FTAG"])
    # x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR"])
    x_train = x_train.fillna(x_train.mean())
    data_normlizer = DataNormalizer()
    x_train = data_normlizer(x_train)
    y_train = data["FTR"]

    classifiers = {
        "KNN": KNeighborsClassifier,
        "DT": DecisionTreeClassifier,
        "RF": RandomForestClassifier,
        "GB": GradientBoostingClassifier,
        "AdaBoost": AdaBoostClassifier,
        "SVC": SVC,
    }
    scores = []
    for clf_name, clf in classifiers.items():
        param_finder = GridSearchCV(
            clf(),
            HYPER_PARAMETERS[clf_name],
            cv=StratifiedShuffleSplit(n_splits=10),
            return_train_score=True,
            verbose=2 ** 31,
            n_jobs=-1,
            refit=True
        )
        param_finder.fit(x_train, y_train)
        params, score = param_finder.best_params_, param_finder.best_score_
        chosen_clf = param_finder.best_estimator_

        clf_results = pd.concat([pd.DataFrame(param_finder.cv_results_["params"]),
                                 pd.DataFrame(param_finder.cv_results_["mean_test_score"], columns=["Accuracy"])],
                                axis=1)
        clf_results = clf_results.sort_values(by='Accuracy', ascending=False)
        clf_results.to_excel(f"{clf_name}_results.xlsx")
        scores.append((chosen_clf, params, score))
    print(scores)
    return sorted(scores, key=lambda c: c[2], reverse=True)[0]


if __name__ == '__main__':
    find_best_model()
    select_best_k_features()
