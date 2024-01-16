import pandas as pd
import numpy as np
from itertools import combinations

class athlete:
    def __init__(self, Competitor, Gender, Country, IndividualDateRank, value):
        self.Competitor = Competitor
        self.Gender = Gender
        self.Country = Country
        self.IndividualDateRank = IndividualDateRank
        self.value = value

    def __str__(self):
        return "{}".format(self.Competitor)
    
    def __str__(self):
        return str(self.Competitor)

class team:
    '''
    team holds a list of athletes
    '''
    def __init__(self, country:str):
        self.country = country 
        self.team = []

    def select_team(self, num_members=4):
        self.team = sorted(self.team, key=lambda athlete: athlete.value, reverse=True)
        if len(self.team) >= num_members:
            self.team = self.team[:num_members] # pick the top 4 members
        else: self.team = self.team[0]

    def new_member(self, member:athlete):
        self.team.append(member)

    def __str__(self):
        return "{}".format(self.team)
    
    def __len__(self):
        return len(self.team)
    
class US_team_combs:
    '''
    function for holding all team combinations to test - use like an iterable object

    main function is `generate_all_combinations`
    '''
    def __init__(self):
        self.index = 0
        # a list of team objects - each has a tuple of athlete objects
        self.team_combs = []

    def generate_all_combinations(self, gender:str, new_dat:pd.DataFrame, events:set, country:str = "USA"):
        '''
        new_dat should be the base data
        '''
        # event_vals = dict()
        # for item in list(new_dat["Apparatus"].unique()): 
        #     event_vals[item] = []
        #     for athl in list(new_dat["Competitor"].unique()): 
        #         score = new_dat.loc[(new_dat["Competitor"] == athl) & (new_dat["Apparatus"] == item), "Score"].mean()
        #         indiv_date_rank = dat.loc[(dat["Country"] == country) & (dat["Competitor"] == athl), "IndividualDateRank"].max() + 1
        #         profile = athlete(athl, gender, country, indiv_date_rank, score)
        #         event_vals[item].append(profile)
        #         # print(profile)
        # for event in event_vals:
        #     event_vals[event] = sorted(event_vals[event], key=lambda ath: ath.value)
        #     # try the top 7 scorers from each event
        #     event_vals[event] = event_vals[event][:7]
        #     event_vals[event] = [i.Competitor for i in event_vals[event]]
        # # handling top scorers in multiple categories?
        # if 
        new_dat = new_dat.loc[new_dat["Gender"] == gender]
        base_athls = team(country)
        for athl in list(new_dat["Competitor"].unique()):
            score = get_value(athl, new_dat, events)
            individual_date_rank = new_dat.loc[(new_dat["Country"] == country) & (new_dat["Competitor"] == athl), "IndividualDateRank"].max() + 1
            base_athls.new_member(athlete(athl, gender, country, individual_date_rank, score))
        base_athls.select_team(num_members=14)
        self.team_combs = list(combinations(base_athls.team, 4))
        
    # define iterative behavior for the whole class
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.team_combs): raise StopIteration
        value = self.team_combs[self.index]
        self.index += 1
        return value
    
# Global Variables
men_events = {"VT", "VT1", "VT2", "SR", "PH", "PB", "HB"}
women_events = {"BB", "VT", "VT1", "VT2", "FX", "UB"}

def get_value(competitor, data, events):
    dat = data.loc[data["Competitor"] == competitor]
    mean_scores = np.zeros(len(events))
    i = 0
    for event in events: 
        # if athlete competes on given event, add to mean score
        if len(dat.loc[dat["Apparatus"] == event]) > 0: mean_scores[i] = dat.loc[dat["Apparatus"] == event, "Score"].mean()
        else: mean_scores[i] = 0
        i += 1
    return np.mean(mean_scores)

def pick_bench_teams(gender:str, events:set, dat:pd.DataFrame):
    teams = []
    for country in list(dat["Country"].unique()):
        # print(country)
        # if country == "USA": continue
        new_team = team(country=country)
        for athl in list(dat.loc[dat["Country"] == country, "Competitor"].unique()):
            # print("\t",athl)
            score = get_value(athl, dat, events)
            # print("\t", score)
            indiv_date_rank = dat.loc[(dat["Country"] == country) & (dat["Competitor"] == athl), "IndividualDateRank"].max() + 1
            new_team.new_member(athlete(Competitor=athl, Gender=gender, Country=country, IndividualDateRank=indiv_date_rank, value=score))
        new_team.select_team()
        teams.append(new_team)
    return teams

def get_teams(data):
    w_benchmarks = pick_bench_teams("w", women_events, data.loc[data["Gender"] == "w"])
    m_benchmarks = pick_bench_teams("m", men_events, data.loc[data["Gender"] == "m"])
    w_USA_teams = US_team_combs()
    w_USA_teams.generate_all_combinations(gender="w", new_dat=data, events=women_events, country="USA")
    m_USA_teams = US_team_combs()
    m_USA_teams.generate_all_combinations(gender="m", new_dat=data, events=men_events, country="USA")

    return w_benchmarks, m_benchmarks, w_USA_teams.team_combs, m_USA_teams.team_combs