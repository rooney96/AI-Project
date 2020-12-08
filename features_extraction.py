import time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import itertools
from constants import *


def _get_players_stats(file_name, urls, teams):
    data = []
    for url in urls:
        option = Options()
        option.headless = False
        driver = webdriver.Chrome(options=option,
                                  executable_path="C:\\Users\\USER\\Downloads\\chromedriver\\chromedriver.exe")
        driver.get(url)
        element = driver.find_element_by_xpath('//*[@id="columns"]/div[1]/div/div[1]/input')
        fifa_stats_list = ["PAC", "DEF", "PHY", "BP", "Sprint Speed"]
        for fifa_stat in fifa_stats_list:
            element.send_keys(fifa_stat)
            element.send_keys(Keys.ENTER)
        element = driver.find_element_by_xpath('//*[@id="columns"]/div[2]/button')
        driver.execute_script("arguments[0].click();", element)
        time.sleep(1)

        element = driver.find_element_by_xpath('//*[@id="adjust"]/div/div[2]/form/div[1]/div[5]/div/div[1]/input')
        for team in teams:
            element.send_keys(team)
            element.send_keys(Keys.ENTER)

        element = driver.find_element_by_xpath('//*[@id="adjust"]/div/div[2]/form/div[1]/div[18]/button')
        driver.execute_script("arguments[0].click();", element)
        time.sleep(1)

        first_iteration = True
        while True:

            table = driver.find_element_by_xpath('//*[@id="adjust"]/div/div[1]/div/table')
            rows = table.find_elements_by_tag_name("tr")
            players_num = len(rows)

            for idx in range(2, players_num):
                table = driver.find_element_by_xpath('//*[@id="adjust"]/div/div[1]/div/table')
                rows = table.find_elements_by_tag_name("tr")
                col = rows[idx].find_elements_by_tag_name('td')
                features = [col[i].text.split("\n")[0] for i in [1, 3, 4, 5, 6, 7, 9, 11, 12, 13]]
                data.append(features)

            try:
                next_button = driver.find_element_by_xpath(
                    '//*[@id="adjust"]/div/div[1]/div/div/a/span[1]') if first_iteration else driver.find_element_by_xpath(
                    '//*[@id="adjust"]/div/div[1]/div/div/a[2]/span[1]')
            except:
                break
            first_iteration = False
            next_button.click()
            time.sleep(3)

        driver.quit()

    columns = ["Name", "Overall Rating", "Potential", "Team", "Pos", "value", "Speed", "PAC", "DEF", "PHY"]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(file_name)


def get_premier_league_lineups(starting_idx: int, ending_idx: int, excel_file_name: str):
    option = Options()
    option.headless = True
    driver = webdriver.Chrome(options=option,
                              executable_path="C:\\Users\\USER\\Downloads\\chromedriver\\chromedriver.exe")
    data = []
    for match in range(starting_idx, ending_idx):
        try:
            home_team_starting = []
            away_team_starting = []
            driver.get("https://www.premierleague.com/match/" + str(match))
            time.sleep(5)
            home_team = driver.find_element_by_xpath(
                "//*[@id=\"mainContent\"]/div/section[2]/div[2]/section/div[3]/div/div/div[1]/div[1]/a[2]/span[1]").text
            away_team = driver.find_element_by_xpath(
                "//*[@id=\"mainContent\"]/div/section[2]/div[2]/section/div[3]/div/div/div[1]/div[3]/a[2]/span[1]").text
            (driver.find_element_by_xpath(
                "//*[@id=\"mainContent\"]/div/section[2]/div[2]/div[2]/div[1]/div/div/ul/li[2]")).click()
            time.sleep(3)
            home_team_formation = driver.find_element_by_xpath(
                "//*[@id=\"mainContent\"]/div/section[2]/div[2]/div[2]/div[2]/section[2]/div/div/div[1]/div/header/div/strong").text
            away_team_formation = driver.find_element_by_xpath(
                '//*[@id="mainContent"]/div/section[2]/div[2]/div[2]/div[2]/section[2]/div/div/div[3]/div/header/div/strong').text

            home_lineup = driver.find_elements_by_class_name("startingLineUpContainer")
            home_team_players_info = list(
                itertools.chain.from_iterable([e.find_elements_by_class_name("info") for e in home_lineup[:4]]))
            away_team_players_info = list(
                itertools.chain.from_iterable([e.find_elements_by_class_name("info") for e in home_lineup[8:12]]))

            for player in home_team_players_info:
                home_team_starting.append(player.find_element_by_class_name("name").text.split('\n')[0])

            for player in away_team_players_info:
                away_team_starting.append(player.find_element_by_class_name("name").text.split('\n')[0])

            data.append(
                [
                    home_team, away_team, home_team_formation, away_team_formation, home_team_starting,
                    away_team_starting
                ]
            )
        except:
            data.append([starting_idx, "empty", "empty", "empty", "empty", "empty"])

    columns = ["HomeTeam", "AwayTeam", "HomeFormation", "AwayFormation", "HomeLineUp", "AwayLineUp"]
    df = pd.DataFrame(data=data, columns=columns)
    df.to_excel(excel_file_name)


if __name__ == '__main__':

    for name, _urls, _teams in zip(TEAM_PER_LEAGUE.keys(), players_stats.values(), TEAM_PER_LEAGUE.values()):
        _get_players_stats(name, _urls, _teams)

    for name, idx in zip(excel_file_name, matches_indexes):
        get_premier_league_lineups(idx[0], idx[1], name)
