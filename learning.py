import pandas as pd
from features_predictions import DataNormalizer
from constants import HYPER_PARAMETERS
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif


selected_features = pd.read_excel('postprocessing_data\\selected_feature.xlsx')
predicted_features: pd.DataFrame = pd.read_excel("collected_data\\match_statistics\\LastThreeSeasons.xlsx").iloc[40:]
predicted_features = predicted_features.reset_index(drop=True)
predicted_features = predicted_features.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"])
data = pd.concat([selected_features, predicted_features], axis=1, join='inner')


def select_best_k_features():
    knn = KNeighborsClassifier(n_neighbors=89, weights='distance')
    random_forest = RandomForestClassifier(criterion='gini', max_depth=7, min_samples_split=4, n_estimators=15)
    decision_tree = DecisionTreeClassifier(max_depth=5, splitter='random')
    svc = SVC(C=0.8, degree=1, gamma='auto', kernel='linear', probability=False)
    gb = GradientBoostingClassifier(max_depth=6, n_estimators=20)
    ab = AdaBoostClassifier(n_estimators=25)
    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR"])
    # filling missing value
    x_train = x_train.fillna(x_train.mean())
    # normalization
    data_normlizer = DataNormalizer()
    x_train = data_normlizer(x_train)
    y_train = data["FTR"]
    features_len = len(x_train.columns)

    pd_columns = ['K', 'KNN acc', 'RF acc', 'DT acc', 'SVC Acc', 'GB Acc', 'AB Acc', 'Best Acc']
    pd_data = []
    for k in range(10, features_len):
        best_features = SelectKBest(f_classif, k=k)
        fit = best_features.fit(x_train, y_train)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(x_train.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        selected_features = (featureScores.nlargest(k, 'Score'))['Specs']
        x_new = x_train[selected_features]

        results = []
        for clf in [knn, random_forest, decision_tree, svc, gb, ab]:
            clf.fit(x_new, y_train)
            scores = cross_val_score(clf, x_new, y_train, cv=10)
            results.append(scores.mean())

        best_acc = sorted(results)[-1]
        pd_data.append([k] + results + [best_acc])

    df = pd.DataFrame(data=pd_data, columns=pd_columns)
    df = df.sort_values(by='Best Acc', ascending=False)
    df.to_excel('learning_results_after_normalization\\SelectKBestResults.xlsx')


def find_best_model():
    x_train = data.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR"])
    x_train = x_train.fillna(x_train.mean())
    data_normalizer = DataNormalizer()
    x_train = data_normalizer(x_train)
    y_train = data["FTR"]

    classifiers = {
        # "KNN": KNeighborsClassifier,
        # "DT": DecisionTreeClassifier,
        # "RF": RandomForestClassifier,
        "GB": GradientBoostingClassifier,
        # "AdaBoost": AdaBoostClassifier,
        # "SVC": SVC,
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

    return sorted(scores, key=lambda c: c[2], reverse=True)[0]


if __name__ == '__main__':
    find_best_model()
    select_best_k_features()
