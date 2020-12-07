from sklearn.metrics import accuracy_score
from constants import TEAMS_LIST
from sklearn.svm import SVC
from features_predictions import *
from constants import FINAL_FEATURES, CREATED_FEATURES, ORIGINAL_FEATURES


def input_is_valid(command: str):
    teams = command.split("vs")
    teams = [team.strip(' ') for team in teams]
    if len(teams) != 2:
        print("Invalid Input")
        return
    if not all(team in TEAMS_LIST for team in teams):
        print("Invalid Teams Name")
        return
    return teams[0], teams[1]


def print_result(home_team, away_team, res):
    if res == 'H':
        print(home_team)
    elif res == 'A':
        print(away_team)
    else:
        print("Draw")


if __name__ == '__main__':

    model = torch.load('final_network_model')
    data = pd.read_excel('learning_ai_algorithms_data\\training_data.xlsx')
    data_normlizer = DataNormalizer()
    data = data_normlizer(data)
    x_train = data[FINAL_FEATURES]
    x_train = x_train.fillna(x_train.mean())
    y_train = data["FTR"]

    clf = SVC(C=0.8, degree=1, gamma='auto', kernel='linear', probability=False)
    clf.fit(x_train, y_train)

    selected_features = pd.read_excel('postprocessing_data\\selected_feature_all_data.xlsx')
    selected_features = selected_features.iloc[1099:]
    selected_features = selected_features.reset_index(drop=True)
    y_test = selected_features["FTR"]
    test = selected_features.drop(columns=["FTR", "Date", "HomeTeam", "AwayTeam", "Unnamed: 0"])
    test = test.fillna(test.mean())
    test = data_normlizer(test)
    print(selected_features[["HomeTeam", "AwayTeam"]])
    x_test = []
    test_cols = [element for element in CREATED_FEATURES + ORIGINAL_FEATURES if
                 element not in ["FTR", "Date", "HomeTeam", "AwayTeam"]]
    for i in range(380):
        features = torch.tensor(test.iloc[i])
        new_features = model(features)
        x_test.append(torch.cat((features, new_features), 0).tolist())

    df = pd.DataFrame(data=x_test, columns=test_cols)
    df = df[FINAL_FEATURES]

    y_pred = clf.predict(df)
    acc = accuracy_score(y_test, y_pred)

    print("please enter home team vs away team:\n"
          "for example: Chelsea vs Arsenal\n"
          "for the team list please enter h for quit enter q:")
    while True:
        command = input()
        if command == "h":
            print(*TEAMS_LIST, sep="\n")
        elif command == "q":
            break
        else:
            if input_is_valid(command) is None:
                continue
            home_team, away_team = input_is_valid(command)
            index = selected_features.index[
                (selected_features['HomeTeam'] == home_team) & (selected_features["AwayTeam"] == away_team)].tolist()
            print_result(home_team, away_team, y_pred[index[0]])
