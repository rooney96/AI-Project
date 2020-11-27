import difflib
from typing import List
import pandas
from constants import *


class FeaturesPreprocessor:
    def __init__(self) -> None:
        self.match_statistics = pandas.read_excel(
            'C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\LastThreeSeasons.xlsx')
        self.lineups17 = pandas.read_excel(
            "C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\LineUp16-17.xlsx")
        self.lineups18 = pandas.read_excel(
            "C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\LineUp17-18.xlsx")
        self.lineups19 = pandas.read_excel(
            "C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\LineUp18-19.xlsx")
        self.lineups20 = pandas.read_excel(
            "C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\LineUp19-20.xlsx")

        self.players17 = pandas.read_excel(
            'C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\Players16-17.xlsx')
        self.players18 = pandas.read_excel(
            'C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\Players17-18.xlsx')
        self.players19 = pandas.read_excel(
            'C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\Players18-19.xlsx')
        self.players20 = pandas.read_excel(
            'C:\\Users\\USER\\PycharmProjects\\PredectingPLMatchesResult\\Datasets\\Players19-20.xlsx')

    def get_all_matches_features(self):
        data = []
        for i in range(41, 1140):
            match_feature = \
                self._get_match_features(
                    self.match_statistics.at[i, "HomeTeam"],
                    self.match_statistics.at[i, "AwayTeam"],
                    self.match_statistics.at[i, "Date"]
                )
            data.append(match_feature)

        pandas.DataFrame(data, columns=CREATED_FEATURES).to_excel("selected_feature.xlsx")

    def update_players_file(self, create_new_file: bool):
        self.players17["Team"] = self.players17["Team"].map(update_players_teams)
        self.players18["Team"] = self.players18["Team"].map(update_players_teams)
        self.players19["Team"] = self.players19["Team"].map(update_players_teams)
        self.players20["Team"] = self.players20["Team"].map(update_players_teams)

        if create_new_file:
            self.players17.to_excel("NewPlayers16-17.xlsx")
            self.players18.to_excel("NewPlayers17-18.xlsx")
            self.players19.to_excel("NewPlayers18-19.xlsx")
            self.players20.to_excel("NewPlayers19-20.xlsx")

    def update_last_three_season_file(self):

        self.match_statistics["HomeTeam"] = t.match_statistics["HomeTeam"].apply(update_match_statistics_teams)
        self.match_statistics["AwayTeam"] = t.match_statistics["AwayTeam"].apply(update_match_statistics_teams)
        self.match_statistics["HomeTeam"] = t.match_statistics["HomeTeam"].apply(update_players_teams)
        self.match_statistics["AwayTeam"] = t.match_statistics["AwayTeam"].apply(update_players_teams)

    def _get_match_features(self, home_team: str, away_team: str, date: str):
        direct_matches: pandas.DataFrame = self._get_direct_games(home_team, away_team, date)
        last_3_away_games = self._get_last_k_team_games(3, away_team, date, False)
        last_3_home_games = self._get_last_k_team_games(3, home_team, date, True)
        home_team_players = self._get_players_stats(home_team, away_team, date.strftime("%d/%m/%Y"), True)
        away_team_players = self._get_players_stats(home_team, away_team, date.strftime("%d/%m/%Y"), False)

        home_away_match_stats = []
        for matches in zip(self._get_last_three_home_matches_statistics(last_3_home_games),
                           self._get_last_three_away_matches_statistics(last_3_away_games)):
            home_away_match_stats += (list(matches))

        match_features = [date.strftime("%d/%m/%Y"), home_team, away_team] \
                         + home_away_match_stats + self._get_direct_matches_statistics(direct_matches) \
                         + self._get_players_statistics(home_team_players, away_team_players) \
                         + [self._get_final_result(home_team, away_team, date)]
        return match_features

    def _get_last_three_home_matches_statistics(self, last_three_matches):
        return [
            last_three_matches["FTHG"].mean(),
            last_three_matches["FTAG"].mean(),
            last_three_matches["HS"].mean(),
            last_three_matches["HF"].mean(),
            last_three_matches["HC"].mean(),
            last_three_matches["HY"].mean(),
            last_three_matches["HR"].mean(),
            self._calculate_points_earned(last_three_matches["FTR"], True),

        ]

    def _get_last_three_away_matches_statistics(self, last_three_matches):
        return [
            last_three_matches["FTAG"].mean(),
            last_three_matches["FTHG"].mean(),
            last_three_matches["AS"].mean(),
            last_three_matches["AF"].mean(),
            last_three_matches["AC"].mean(),
            last_three_matches["AY"].mean(),
            last_three_matches["AR"].mean(),
            self._calculate_points_earned(last_three_matches["FTR"], False),

        ]

    def _get_direct_matches_statistics(self, direct_matches):
        return [
            self._calculate_points_earned(direct_matches["FTR"], True) if not direct_matches.empty else 0,
            self._calculate_points_earned(direct_matches["FTR"], False) if not direct_matches.empty else 0,
            direct_matches["FTHG"].mean() if not direct_matches.empty else 0,
            direct_matches["FTAG"].mean() if not direct_matches.empty else 0,
            direct_matches["HR"].mean() if not direct_matches.empty else 0,
            direct_matches["AR"].mean() if not direct_matches.empty else 0,

        ]

    @staticmethod
    def _get_players_statistics(home_team_players, away_team_players):
        return [
            home_team_players.loc[home_team_players["Pos"].apply(lambda p: p in ATTACK)]["PAC"].mean(),
            away_team_players.loc[away_team_players["Pos"].apply(lambda p: p in ATTACK)]["PAC"].mean(),
            home_team_players.loc[home_team_players["Pos"].apply(lambda p: p in DEFENCE)]["PAC"].mean(),
            away_team_players.loc[away_team_players["Pos"].apply(lambda p: p in DEFENCE)]["PAC"].mean(),
            home_team_players["PHY"].mean(),
            away_team_players["PHY"].mean(),
            home_team_players.loc[home_team_players["Pos"].apply(lambda p: p in ATTACK)]["Overall Rating"].mean() -
            away_team_players.loc[away_team_players["Pos"].apply(lambda p: p in DEFENCE)]["Overall Rating"].mean(),
            away_team_players.loc[away_team_players["Pos"].apply(lambda p: p in ATTACK)]["Overall Rating"].mean() -
            home_team_players.loc[home_team_players["Pos"].apply(lambda p: p in DEFENCE)]["Overall Rating"].mean(),
            home_team_players.loc[home_team_players["Pos"].apply(lambda p: p in MID)]["Overall Rating"].mean() -
            away_team_players.loc[away_team_players["Pos"].apply(lambda p: p in ATTACK)]["Overall Rating"].mean(),

        ]

    def _get_last_k_team_games(self, k: int, team: str, date: str, at_home: bool):

        team_features = self.match_statistics[
            self.match_statistics["HomeTeam"] == team] if at_home else self.match_statistics[
            self.match_statistics["AwayTeam"] == team]

        team_features = team_features.reset_index(drop=True)
        index = team_features[(team_features['Date'] == date)].index.tolist()
        index = index.pop() if len(index) != 0 else 0
        k = k if k < len(team_features['Date']) else len(team_features['Date'])
        return team_features.iloc[index - k if index - k > 0 else 0: index]

    def _get_direct_games(self, home_team: str, away_team: str, date):
        direct_games = self.match_statistics[
            ((self.match_statistics["AwayTeam"] == away_team) & (self.match_statistics["HomeTeam"] == home_team))]
        direct_games = direct_games.reset_index(drop=True)
        index = direct_games[(direct_games['Date'] == date)].index.tolist()
        index = index.pop() if len(index) != 0 else 0
        return direct_games.iloc[: index]

    def _get_lineups_by_season(self, date: str):
        year = int((date.split('/'))[-1]) - 2000
        month = int((date.split('/'))[1])
        year += month >= 7
        switch = {
            17: self.lineups17,
            18: self.lineups18,
            19: self.lineups19,
            20: self.lineups20,

        }
        return switch[year]

    def _get_players_stats_by_season(self, date: str, after_t_w: bool):
        year = int((date.split('/'))[-1]) - 2000
        month = int((date.split('/'))[1])
        year += month >= 7
        year += after_t_w
        year = 20 if year > 20 else year
        switch = {
            17: self.players17,
            18: self.players18,
            19: self.players19,
            20: self.players20,

        }
        return switch[year]

    @staticmethod
    def _calculate_points_earned(results: List[str], is_home: bool):
        points = 0
        for result in results:
            points += (result == "D")
            points += 3 * (result == "H") if is_home else 3 * (result == "A")
        return points

    def _get_final_result(self, home_team, away_team, date):
        game_data = self.match_statistics[
            (self.match_statistics["HomeTeam"] == home_team) & (self.match_statistics["AwayTeam"] == away_team) & (
                    self.match_statistics["Date"] == date)]
        game_data = game_data.reset_index(drop=True)
        return game_data.at[0, "FTR"]

    def _get_match_line_up(self, home_team, away_team, date, at_home):
        lineup_by_season = self._get_lineups_by_season(date)
        match = lineup_by_season[
            (lineup_by_season["HomeTeam"] == home_team) & (lineup_by_season["AwayTeam"] == away_team)]
        team = match.iloc[0]["HomeLineUp"] if at_home else match.iloc[0]["AwayLineUp"]
        return [e.replace('\'', '').strip() for e in team[1: -1].split(",")]

    def _get_team_formation(self, home_team, away_team, date, at_home):
        lineup_by_season = self._get_lineups_by_season(date)
        match = lineup_by_season[
            (lineup_by_season["HomeTeam"] == home_team) & (lineup_by_season["AwayTeam"] == away_team)]
        return match.iloc[0]["HomeFormation"] if at_home else match.iloc[0]["AwayFormation"]

    def _get_players_stats(self, home_team, away_team, date, at_home):
        data = []
        for p in self._get_match_line_up(home_team, away_team, date, at_home):
            players = self._get_players_stats_by_season(date, False)
            cols = list(players.columns.values)[1:]
            team_players = players[(players["Team"] == home_team)] if at_home else players[
                (players["Team"] == away_team)]
            try:
                if p == "Zanka":
                    p = "M. Jørgensen"
                closest_match = difflib.get_close_matches(p, list(team_players["Name"]), cutoff=0.3)[0]
                data.append((team_players[team_players["Name"] == closest_match]).iloc[0])
            except:
                continue
        return pandas.DataFrame(data=data, columns=cols).reset_index(drop=True)


def update_match_statistics_teams(team):
    return MATCH_STATS_FILE_CHANGES[team] if team in MATCH_STATS_FILE_CHANGES.keys() else team


def update_players_teams(team):
    return PLAYERS_FILE_CHANGES[team] if team in PLAYERS_FILE_CHANGES.keys() else team


if __name__ == '__main__':
    t = FeaturesPreprocessor()
    t.get_all_matches_features()

    '''
    Getting input from user
    home_team, away_team, date = input("please enter home team vs away team and the date of the game\n").split(",")
    res = t.get_match_features(home_team, away_team, date)
    print(res)
    '''

    '''
    Get only numerical features
    numerical_features = new_features.drop(columns=["Date", "HomeTeam", "AwayTeam", "FTR"])
    print(numerical_features)
    print(numerical_features)
    '''

    '''
    Drop Nan Values
    numerical_features = numerical_features.dropna()
    '''

    '''
    returns columns names
    col = sorted(numerical_features)
    '''

    '''
    normalizing data
    x = preprocessing.normalize(numerical_features)
    print(pandas.DataFrame(x, columns=col))
    '''

    '''
    Print all data frame
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd)
    '''
