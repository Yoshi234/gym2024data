'''
simulate results to find optimum team composition for 

picking benchmark teams: 

sim steps
(1) qualifying --- simulate qualifying stage for teams 
(2) team finals ---
(3) individual all around finals ---
(4) 

For picking potential USA candidates
- pick the top 20 individuals in terms of average score for each event, 
and try all of the combinations available from those team members
'''
import math
import pandas as pd
import numpy as np
from simulator_help.pick_teams import get_teams
import pickle
from modeling.training_preprocess.preprocess import preprocess_sample
from simulator_help.pick_teams import men_events, women_events

def load_model(folder, mod_name):
    filename = "{}/{}.pkl".format(folder, mod_name)
    print("reading from {}".format(filename))
    with open(filename, "rb") as f:
        model = pickle.load(f)
    with open("{}/{}_features.pkl".format(folder, mod_name), "rb") as f2:
        features = pickle.load(f2)
    return model, features

def run_r6(athlete, round, apparatus, mean, std, r6_model, r6_features):
    '''
    takes as input, an `athlete` object type, and outputs the expected score
    for a given set of parameters

    `athlete` is defined in `simulator_help/pick_teams.py`
    '''
    vals = np.zeros(len(r6_features))
    features = pd.DataFrame([vals], columns=r6_features)
    x = features
    
    name = athlete.Competitor   
    # replace spaces with underscores
    name = name.replace(' ', '_')

    x.iloc[0]['category__Competitor_{}'.format(name)] = 1.0
    x.iloc[0]['category__Country_{}'.format(athlete.Country)] = 1.0
    x.iloc[0]['category__Apparatus_{}'.format(apparatus)] = 1.0
    x.iloc[0]['category__Round_{}'.format(round)] = 1.0
    x.iloc[0]['log0__IndividualDateRank'] = math.log(athlete.IndividualDateRank)
    x.iloc[0]['category__Gender_{}'.format(athlete.Gender)]

    x = x[r6_features]
    
    y = r6_model.predict(features)
    y = y[0]
    y = y*std + mean
    return y

# get scaling factors for re-scaling data
def scale_output(data):
    mu = data["Score"].mean()
    sd = data["Score"].std()
    return mu, sd

class entry:
    def __init__(self, country, name, event, score):
        self.country = country
        self.name = name
        self.event = event
        self.score = score

def run_qualifying(benchmarks, usa, events, mu, sd, model, features):
    quali_results = pd.DataFrame(columns=["country", "name", "event", "score"])
    for team in benchmarks:
        if len(team) == 4: 
            for athl in team.team:
                for apparatus in events:
                    score = run_r6(athl, "qual", apparatus, mu, sd, model, features)
                    vals = [athl.Country, athl.Competitor, apparatus, score]
                    entry = pd.DataFrame([vals], columns=["country", "name", "event", "score"])
                    quali_results = pd.concat([quali_results, entry], ignore_index=True)
        else: 
            athl = team.team[0]
            score = run_r6(athl, "qual", apparatus, mu, sd, model, features)
            vals = [athl.Country, athl.Competitor, apparatus, score]
            entry = pd.DataFrame([vals], columns=["country", "name", "event", "score"])
            quali_results = pd.concat([quali_results, entry], ignore_index=True)
    for athl in usa:
        for apparatus in events:
            score = run_r6(athl, "qual", apparatus, mu, sd, model, features)
            vals = [athl.Country, athl.Competitor, apparatus, score]
            entry = pd.DataFrame([vals], columns=["country", "name", "event", "score"])
            quali_results = pd.concat([quali_results, entry], ignore_index=True)
    return quali_results

class result:
    def __init__(self, score, country):
        self.score = score
        self.country = country

def get_qualifying_teams(quali_results):
    '''
    takes df of results and returns qualifying team
    '''
    qualifying_teams = []
    for country in list(quali_results["country"].unique()):
        score = quali_results.loc[quali_results["country"] == country, "score"].sum()
        qualifying_teams.append(result(score, country))
    qualifying_teams = sorted(qualifying_teams, key=lambda res: res.score, reverse=True)
    qualifying_teams = qualifying_teams[:8]
    qualifying_teams = [result.country for result in qualifying_teams]
    return qualifying_teams

def run_team_final(quali_results, benchmarks, usa, events, mu, sd, model, features):
    team_final_results = pd.DataFrame(columns=["country", "name", "event", "score"])
    qualifying_teams = get_qualifying_teams(quali_results)
    for team in benchmarks:
        if team.country in qualifying_teams:
            for athl in team.team:
                for apparatus in events:
                    score = run_r6(athl, "TeamFinal", apparatus, mu, sd, model, features)
                    vals = [athl.Country, athl.Competitor, apparatus, score]
                    entry = pd.DataFrame([vals], columns=["country", "name", "event", "score"])
                    team_final_results = pd.concat([team_final_results, entry], ignore_index=True)
        else: continue
    for athl in usa:
        for apparatus in events:
            score = run_r6(athl, "qual", apparatus, mu, sd, model, features)
            vals = [athl.Country, athl.Competitor, apparatus, score]
            entry = pd.DataFrame([vals], columns=["country", "name", "event", "score"])
            team_final_results = pd.concat([team_final_results, entry], ignore_index=True)
    return team_final_results

def get_team_finals_medals(team_finals_results, usa):
    ranks = []
    for country in list(team_finals_results["country"].unique()):
        score = team_finals_results.loc[team_finals_results["country"] == country, "score"].sum()
        ranks.append(result(score, country))
    ranks = sorted(ranks, key=lambda res: res.score, reverse=True)
    ranks = [rank.country for rank in ranks]
    for i in range(len(ranks)): 
        if ranks[i].country == "USA" and i <= 2: 
            if i == 0: print("USA got gold in team final")
            elif i == 1: print("USA got silver in team final")
            elif i == 2: print("USA got bronze in team final")
            print("team used: ")
            for i in usa: print("\t", i)
        else: continue

def run_sim(mu, sd, benchmarks, usa, events, model, features):
    # run quali - see which scores make it or not
    # if on team
    quali_results = run_qualifying(benchmarks, usa, events, mu, sd, model, features)
    # run team final - 
    team_final_scores = run_team_final(quali_results, benchmarks, usa, events, mu, sd, model, features)
    get_team_finals_medals(team_final_scores, usa)
    # run individual event final 
    # all_around_final_scores, medals = run_aa_final(quali_results, benchmarks, usa, events, mu, sd, model, features)
    # # run event finals
    # event_final_scores, medals = run_event_finals(quali_results, benchmarks, usa, events, mu, sd, model, features)
    return None

def run_combination_simulations():
    data_file = "processed_data/all_data.csv"    
    dat = pd.read_csv(data_file)

    mu, sd = scale_output(dat)
    print("generating teams")
    w_benchmark, m_benchmark, w_USA, m_USA = get_teams(dat)

    model, features = load_model("modeling/pretrained", "r6_mod")
    
    print('running women sim')
    # run women's simulations
    for w_team in range(len(w_USA)):
        run_sim(mu, sd, w_benchmark, w_team, women_events, model, features)

    print('running men sim')
    for m_team in range(len(m_USA)):
        run_sim(mu, sd, m_benchmark, m_team, men_events, model, features)

if __name__ == "__main__":
    run_combination_simulations()
   