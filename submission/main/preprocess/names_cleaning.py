'''
source file for setting the date features in the data
'''

from datetime import datetime
import pandas as pd
import numpy as np

# fix list

# GLOBAL VARIABLE
fixes = {("li wen yeoh", "rachel yeoh li wen", "li wen rachel yeoh"): "li wen yeoh",
         ("alice d amato", "alice d'amato"): "alice d'amato", 
         ("farah ann abdul hadi", "hadi abdul"): "farah ann abdul hadi", 
         ("elisa hämmerle", "elisa haemmerle"): "elisa hämmerle", 
         ("viktoriia listunova", "viktoria listunova"): "viktoria listunova", 
         ("vera pol", "vera van pol"): "vera van pol", 
         ("pauline schaefer betz", "pauline schaefer-betz"): "pauline schaefer-betz", 
         ("hua tien ting", "hua ting", "hua-tien ting"): "hua tien ting", 
         ("rakan alharith", "rakan al harithi", "rakah al harithi", "al harith rakan", "al-harith rakan"): "rakan al-harith", 
         ("alejandra alvarez diaz", "alejandra sofia alvarez diaz"): "alejandra alvarez diaz", 
         ("mc ewan ateer", "ewan mc ateer", "ewan mcateer"): "ewan mcateer", 
         ("mehmet ayberk", "mehmet ayberk kosak"): "mehmet ayberk", 
         ("gabriel barbosa", "gabriela barbosa"): "gabriela barbosa", 
         ("camil betances", "camil betances reyes"): "camil betances reyes",
         ("valentin brostella", "valentina brostella", "valentina brostella arias"): "valentina brostella", 
         ("jossimar orlando calvo moreno", "jossimar orlando calvo moreno jo"): "jossimar orlando calvo moreno", 
         ("taylor christopulos", "taylor troy christopulos"): "taylor christopulos", 
         ("jorge rubio cerro", "jorge rubio"): "jorge rubio", 
         ("mc rhys clenaghan", "rhys mc clenaghan", "rhys mcclenaghan"): "rhys mcclenaghan", 
         ("matt cormier", "matthew cormier"): "matthew cormier", 
         ("agust davidsson", "agust ingi davidsson"): "agust davidsson", 
         ("melanie de jesus dos santos", "mélanie de jesus dos santos"): "mélanie de jesus dos santos", 
         ("alejandro de la cruz", "alejandro de la cruz gato"): "alejandro de la cruz", 
         ("fabian de luna", "fabian luna"): "fabian luna", 
         ("loran de munck", "loran munck"): "loran munck", 
         ("martijn de veer", "martijn veer"): "martijn veer", 
         ("laurie denommee", "laurie denommée"): "laurie denommee", 
         ("stella diaz", "stella loren diaz", "stella diaz muniz"): "stella diaz", 
         ("sam dick", "samual dick", "samuel dick"): "samuel dick", 
         ("dusan djordjevic", "dusan dordeivc"): "dusan dordevic", 
         ("ahmed elmaraghy", "ahmed el maraghy"): "ahmed el maraghy", 
         ("ruby evan", "ruby evans"): "ruby evans", 
         ("lea franceries", "léa franceries"): "lea franceries", 
         ("william fu allen", "william fu-allen", "william fuallen"): "william fu-allen", 
         ("luisa maia", "luisa gomes maia"): "luisa gomes maia", 
         ("jermain gruenberg", "jermain grünberg"): "jermain gruenberg", 
         ("hildur gudmundsdottir", "hildur maja gudmundsdottir"): "hildur gudmundsdottir", 
         ("yuri guimaraes", "yuri guimarães"): "yuri guimaraes", 
         ("yunus gundogdu", "yunus emre gundogdu", "yunus gündogdu"): "yunus gundogdu", 
         ("joseph hatoguan", "joseph judah hatoguan"): "joseph hatoguan", 
         ("hillary heron", "hillary heron soto"): "hillary heron", 
         ("ainhoa herrero lugo", "ainhoa sofia herrero lugo"): "ainhoa herrero lugo", 
         ("vinzenz hoeck", "vinzenz höck"): "vinzenz hoeck", 
         ("carlo hoerr", "carlo horr", "carlo hörr"): "carlo hoerr", 
         ("yen chang huang", "yen-chang  huang"): "yen chang huang", 
         ("yuan hsi hung", "yuan-hsi hung"): "yuan hsi hung", 
         ("karl idesjoe", "karl idesjö"): "karl idesjoe", 
         ("yo-seop  jeon", "yoseop jeon"): "yoseop jeon", 
         ("pavel karenejenko", "pavel karnejenko"): "pavel karnejenko", 
         ("mohamad khalil", "mohamed khalil"): "mohamed khalil", 
         ("anna lena koenig", "anna-lena könig"): "anna lena koenig", 
         ("shante koti", "shanté koti"): "shante koti", 
         ("severin kranzlmueller", "severin kranzlmüller"): "severin kranzmueller", 
         ("margret kristinsdottir", "margret lea kristinsdottir"): "margret kristinsdottir", 
         ("carina kroell", "carina kröll"): "carina kroell", 
         ("hansha gayashan kumarasinghege", "hansa gayashan kumarasinghege"): "hansha gayashan kumarasinghege", 
         ("kylee kvamme", "kylee ann kvamme"): "kylee kvamme", 
         ("pin lai", "pin ju lai", "pin-ju lai"): "pin ju lai", 
         ("xingyu lan", "xingyu  lan"): "xingyu lan", 
         ("thanh tung le", "thanh tùng  lê"): "thanh tung le", 
         ("cassandra lee", "cassandra paige lee"): "cassandra lee", 
         ("chih lee", "chih-kai lee", "chih kai lee"): "chih lee", 
         ("yi chun liao", "yi-chun liao"): "yi chun liao", 
         ("chaopan lin", "chaopan  lin"): "chaopan lin", 
         ("guan lin", "guan yi lin", "guan-yi lin"): "guan yi lin", 
         ("yi lin", "yi-chen lin", "yi chen lin"): "yi chen lin", 
         ("sara loikas", "sara sofia loikas"): "sara loikas", 
         ("jorge vega lopez", "jorge vega"): "jorge vega lopez", 
         ("noemie louon", "noémie louon"): "noemie louon", 
         ("julie madsoe", "julie madsø"): "julie madsoe", 
         ("sani maekelae", "sani mäkelä"): "sani maekelae", 
         ("razvan denis marc", "razvan-denis marc"): "razvan denis marc", 
         ("antonia marihuan", "antonia marihuan rubio"): "antonia marihuan rubio", 
         ("clay mason stephens", "clay masonstephens"): "clay mason stephens", 
         ("lorena medina", "lorena medina cobos"): "lorena medina", 
         ("dimitrijs mickevics", "dmitrijs mickevics"): "dimitrijs mickevics", 
         ("toma modoianu zseder", "toma modoianu-zseder"): "toma modoianu zseder", 
         ("alissa moerz", "alissa mörz"): "alissa moerz", 
         ("charlie moerz", "charlize mörz", "charlize moerz"): "charlize moerz", 
         ("sasiwimion mueangphuan", "sasiwimon mueangphuan"): "sasiwimon mueangphuan", 
         ("karla navas", "karla navas boyd"): "karla navas", 
         ("annalise newman achee", "annalise becca newman achee"): "annalise newman achee", 
         ("khanh phong nguyen", "van khanh phong nguyen"): "khanh phong nguyen", 
         ("audrys nin", "audrys nin reyes"): "audrys nin reyes", 
         ("isaac nunez", "isaac nuñez"): "isaac nunez", 
         ("aberdeen o driscoll", "aberdeen o'driscol"): "aberdeen o'driscol", 
         ("dagur olafsson", "dagur kari olafsson"): "dagur olafsson", 
         ("ahmet onder", "ahmet önder"): "ahmet onder", 
         ("ananya patanakul", "ananya belle patanakul"): "ananya patanakul", 
         ("dmitriy patanin", "dmitry patanin"): "dmitry patanin", 
         ("andres josue perez gines", "andres josue perez ginez"): "andres josue perez ginez", 
         ("makarena pinto adasme", "makarena daisy pinto adasme"): "makarena daisy pinto adasme", 
         ("jimi päivänen", "jimi pävänen"): "jimi pävänen", 
         ("lea quaas", "lea marie quaas"): "lea marie quaas", 
         ("larasati regganis", "larasati rengganis"): "larasati rengganis", 
         ("michael reid", "michael james reid"): "michael reid", 
         ("fred richard", "frederick richard"): "frederick richard", 
         ("keira rolston larking", "keira rolston-larking"): "keira rolston-larking", 
         ("leo saladino", "léo saladino"): "leo saladino", 
         ("göksu üctas sanli", "goksu uctas sanli"): "goksu uctas sanli", 
         ("karina schoenmaier", "karina schönmaier"): "karina schoenmaier", 
         ("yu shiao", "yu jan shiao", "yu-jan shiao"): "yu jan shiao", 
         ("poppy grace stickler", "poppy stickler"): "poppy stickler", 
         ("chia hung tang", "chia-hung tang"): "chia hung tang", 
         ("derin tanriyasukur", "derin tanriyasükür"): "derin tanriyasukur", 
         ("jonas thorisson", "jonas ingi thorisson"): "jonas thorisson", 
         ("jan gwynn timbang", "jann gwynn timbang"): "jan gwynn timbang", 
         ("juliane toessebro", "juliane tøssebro"): "juliane toessebro", 
         ("wei tseng", "wei sheng tseng", "wei-sheng tseng"): "wei sheng tseng", 
         ("kim vanstroem", "kim vanstrom", "kim wanström"): "kim vanstrom", 
         ("adria vera", "adria vera mora"): "adria vera", 
         ("daniel villafane", "daniel villafañe", "daniel angel villafane"): "daniel angel villafane", 
         ("max whitlock", "max whitlock obe"): "max whitlock", 
         ("hiu ying wong", "hiu ying angel wong"): "hiu ying wong", 
         ("sing wu", "sing fen wu"): "sing fen wu", 
         ("ruoteng xiao", "ruoteng  xiao"): "ruoteng xiao", 
         ("korkem yerbossynkyzy", "korkem yerobssynkyzy"): "korkem yerobssynkyzy", 
         ("jin-seong  yun", "jinseong yun"): "jinseong yun", 
         ("sam zakutney", "samuel zakutney"): "samuel zakutney", 
         ("boheng zhang", "boheng  zhang"): "boheng zhang", 
         ("ga-ram  bae", "garam bae"): "garam bae", 
}
# continue from line 142 in the "competitor_fixes.txt" file when fixing the names

def set_date_features():
    pass

def match_cost(x,y,c):
    if x==y: return 0
    else: return c

class sim_string:
    def __init__(self, x:str):
        self.x = x
        self.score = -1

    def __str__(self):
        return str(self.x)

    def __len__(self):
        return len(self.x)
    
    def compare_score(self, other, gc=1, mc=1, min_cost=1):
        '''
        run optimal string alignment for sim_string=self and sim_string=other
        '''
        OPT = dict()
        m = len(self)
        n = len(other)
    
        for i in range(m+1):
            OPT[(i,0)] = i*gc
        for j in range(n+1):
            OPT[(0,j)] = j*gc

        for i in range(1,m+1):
            for j in range(1,n+1):
                OPT[(i,j)] = min(match_cost(self.x[i-1],other.x[j-1],mc) + OPT[(i-1,j-1)],
                                 gc + OPT[(i-1,j)], 
                                 gc + OPT[(i,j-1)])
        self.score = OPT[(m,n)]
        if self.score <= min_cost:
            print("{} v. {} = {}".format(self.x, other.x, self.score))

def find_sim_names():
    data_file = "all_data"
    file_name = "../processed_data/{}.csv".format(data_file)
    df = pd.read_csv(file_name)

    competitors = list(df["Competitor"].unique())

    new_comps = []
    for comp in competitors:
        new_comps.append(sim_string(comp))

    fixes = []
    for i in range(len(competitors)):
        if new_comps[i].x in fixes: continue
        for j in range(i+1,len(competitors)):
            new_comps[i].compare_score(new_comps[j], gc=0.3, mc=5, min_cost=2)
            if new_comps[i].score < 2: fixes.append(new_comps[i].x)
    # print(fixes)

def main():
    find_sim_names()
    


if __name__ == "__main__":
    main()