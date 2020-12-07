import numpy as np

DEFENCE = ["RF", "LF", "RW", "LW", "ST", "CF"]
MID = ["CAM", "LM", "CM", "RM", "CDM", "LWB"]
ATTACK = ["LB", "RB", "CB", "RWB", "LWB"]

CREATED_FEATURES = [
    "Date", "HomeTeam", "AwayTeam", "avg3_home_goals_scored", "avg3_away_goals_scored",
    "avg3_home_goals_conceded", "avg3_away_goals_conceded", "avg3_home_shots", "avg3_away_shots",
    "avg3_home_fouls", "avg3_away_fouls", "avg3_home_corners", "avg3_away_corners", "avg3_home_yellow_cards",
    "avg3_away_yellow_cards", "avg3_home_red_cards", "avg3_away_red_cards", "last3_home_points_earned",
    "last3_away_points_earned", "direct_home_points", "direct_away_points", "avg_direct_home_goals_scored",
    "avg_direct_away_goals_scored", "avg_direct_home_red_cards", "avg_direct_away_red_cards",
    "avg_home_att_pac", "avg_away_att_pac", "avg_home_def_pac", "avg_away_def_pac", "avg_home_phy",
    "avg_away_phy", "home_att_away_def_diff", "away_att_home_def_diff", "home_away_mid_diff", "FTR"
]

ORIGINAL_FEATURES = [
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"
]

FINAL_FEATURES = [
    "HST",
    "AST",
    "home_away_mid_diff",
    "home_att_away_def_diff",
    "away_att_home_def_diff",
    "HS",
    "AS",
    "avg3_away_shots",
    "avg3_home_shots",
    "avg3_away_goals_scored"
]

MATCH_STATS_FILE_CHANGES = {
    "Tottenham": "Tottenham Hotspur",
    "Swansea": "Swansea City",
    "Leicester": "Leicester City",
    "West Brom": "West Bromwich Albion",
    "Cardiff": "Cardiff City",
    "Stoke": "Stoke City",
    "Brighton": "Brighton and Hove Albion",
    "Man City": "Manchester City",
    "Huddersfield": "Huddersfield Town",
    "Wolves": "Wolverhampton Wanderers",
    "Hull": "Hull City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "West Ham": "West Ham United",
}

PLAYERS_FILE_CHANGES = {
    "Bournemouth": "AFC Bournemouth",
    "Brighton & Hove Albion": "Brighton and Hove Albion"
}

excel_file_name = [
    "LineUp16-17.xlsx",
    "LineUp17-18.xlsx",
    "LineUp18-19.xlsx",
    "LineUp19-20.xlsx",
]

matches_indexes = [
    [14040, 14420],
    [22342, 22722],
    [38308, 38688],
    [46605, 46985]
]

players_stats = {
    "Players16-17.xlsx": [
        "https://sofifa.com/players?type=all&r=160049&set=true",
        "https://sofifa.com/players?type=all&r=170007&set=true",
        "https://sofifa.com/players?type=all&r=170042&set=true",
    ],
    "Players17-18.xlsx": [
        "https://sofifa.com/players?type=all&r=170082&set=true",
        "https://sofifa.com/players?type=all&r=180013&set=true",
        "https://sofifa.com/players?type=all&r=180047&set=true"
    ],
    "Players18-19.xlsx": [
        "https://sofifa.com/players?type=all&r=180079&set=true",
        "https://sofifa.com/players?type=all&r=190008&set=true",
        "https://sofifa.com/players?type=all&r=190042&set=true"
    ],
    "Players19-20.xlsx": [
        "https://sofifa.com/players?type=all&r=190072&set=true",
        "https://sofifa.com/players?type=all&r=200007&set=true",
        "https://sofifa.com/players?type=all&r=200028&set=true"
    ]
}

TEAM_PER_LEAGUE = {
    "Players16-17.xlsx": [
        'Arsenal', 'Manchester United', 'AFC Bournemouth', 'Everton', 'Tottenham Hotspur', 'Chelsea', 'Sunderland',
        'Liverpool', 'Leicester City', 'Manchester City', 'Crystal Palace', 'Middlesbrough', 'Swansea City',
        'Stoke City', 'Burnley', 'West Bromwich Albion', 'West Ham United', 'Watford', 'Southampton', 'Hull City'
    ],
    "Players17-18.xlsx": [
        'Arsenal', 'Manchester United', 'AFC Bournemouth', 'Everton', 'Tottenham Hotspur', 'Chelsea', 'Liverpool',
        'Cardiff City', 'Leicester City', 'Manchester City', 'Crystal Palace', 'Brighton and Hove Albion',
        'Burnley', 'Fulham', 'West Ham United', 'Huddersfield Town', 'Wolverhampton Wanderers', 'Newcastle United',
        'Watford', 'Southampton'
    ],
    "Players18-19.xlsx": [
        'Arsenal', 'Manchester United', 'AFC Bournemouth', 'Everton', 'Tottenham Hotspur', 'Chelsea', 'Liverpool',
        'Leicester City', 'Manchester City', 'Crystal Palace', 'Brighton and Hove Albion', 'Swansea City',
        'Stoke City', 'Burnley', 'West Bromwich Albion', 'West Ham United', 'Huddersfield Town', 'Newcastle United',
        'Watford', 'Southampton'
    ],
    "Players19-20.xlsx": [
        'Aston Villa', 'Norwich City', 'Arsenal', 'Manchester United', 'AFC Bournemouth', 'Everton',
        'Sheffield United', 'Tottenham Hotspur', 'Chelsea', 'Liverpool', 'Leicester City', 'Manchester City',
        'Crystal Palace', 'Brighton and Hove Albion', 'Burnley', 'West Ham United', 'Wolverhampton Wanderers',
        'Newcastle United', 'Watford', 'Southampton'
    ]
}

HYPER_PARAMETERS = {
    "KNN": {
        'n_neighbors': [i for i in range(3, 20) if i % 2 != 0],
        'weights': ['uniform', 'distance']
    },

    "DT": {
        'max_depth': [i for i in range(5, 20)],
        'splitter': ['best', 'random']
    },

    "RF": {
        'n_estimators': [i for i in range(5, 16)],
        'criterion': ['gini'],
        'max_depth': range(5, 15),
        'min_samples_split': [pow(2, i) for i in range(2, 4)]
    },

    "SVC": {
        'C': [i / 10 for i in range(1, 11)],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [i for i in range(1, 5)],
        'probability': [True, False],
        'gamma': ['scale', 'auto']
    },

    "GB": {
        'min_samples_leaf': np.linspace(2e-4, 0.01, num=7),
        'min_samples_split': np.linspace(2e-4, 0.05, num=7),
        'max_depth': [7],
        'n_estimators': [100, 200, 300, 400],
    },
    "AdaBoost": {
        "n_estimators": [i for i in range(30, 300, 10)],
    }
}

TEAMS_LIST = [
    "Arsenal",
    "Everton",
    "Chelsea",
    "Liverpool",
    "Tottenham Hotspur",
    "Leicester City",
    "Brighton and Hove Albion",
    "Manchester City",
    "Wolverhampton Wanderers",
    "Manchester United",
    "Newcastle United",
    "Norwich City",
    "West Ham United",
    "Aston Villa",
    'AFC Bournemouth',
    'Sheffield United',
    "Crystal Palace",
    "Burnley",
    "Watford",
    "Southampton"
]
