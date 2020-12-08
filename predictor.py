from constants import TEAMS_LIST
from features_predictions import *
from constants import FINAL_FEATURES, CREATED_FEATURES, ORIGINAL_FEATURES
import joblib
import argparse
from sklearn.metrics import accuracy_score


selected_features = pd.read_excel('postprocessing_data\\selected_feature.xlsx')
predicted_features: pd.DataFrame = pd.read_excel("collected_data\\match_statistics\\LastThreeSeasons.xlsx").iloc[40:]
predicted_features = predicted_features.reset_index(drop=True)
predicted_features = predicted_features.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"])
data = pd.concat([selected_features, predicted_features], axis=1, join='inner')


def print_result(home_team, away_team, res):
    if res == 'H':
        print(home_team)
    elif res == 'A':
        print(away_team)
    else:
        print("Draw")


if __name__ == '__main__':

    # model = torch.load('network_model')
    model = DynamicNet(D_in, 256, D_out)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    runNN1(optimizer, model)

    clf = joblib.load('finalized_model.sav')
    # # TODO: make sure its working on windows
    # model = model.double()

    data_normalizer = DataNormalizer()
    selected_features = pd.read_excel('postprocessing_data/selected_feature_all_data.xlsx')
    selected_features = selected_features.iloc[1100:]
    selected_features = selected_features.reset_index(drop=True)
    y_test = selected_features["FTR"]
    test = selected_features.drop(columns=["FTR", "Date", "HomeTeam", "AwayTeam"])
    test = test.fillna(test.mean())
    test = data_normalizer(test)

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
    print(f"{len(y_pred), len(y_test)}")
    score = 0
    for idx in range(len(y_pred)):
        score += y_pred[idx] == y_test[idx]
    print(f"accuracy = {score/len(y_test)}")
    parser = argparse.ArgumentParser(description='Premier League Match Predictor:', usage='predictor.py [options]')
    parser.add_argument('-H', '--home_team', metavar='', help='Home Team')
    parser.add_argument('-A', '--away_team', metavar='', help='Away Team')
    parser.add_argument('-T', action='store_true', help="Displays available teams")

    args = parser.parse_args()
    if args.T:
        print("Premier League Teams List For Season 2019/20:")
        print(*TEAMS_LIST, sep="\n")

    else:
        if not all(team in TEAMS_LIST for team in [args.home_team, args.away_team]):
            print("Invalid Teams Name")
        else:
            index = selected_features.index[
                (selected_features['HomeTeam'] == args.home_team) & (selected_features["AwayTeam"] == args.away_team)].tolist()
            print_result(args.home_team, args.away_team, y_pred[index[0]])
