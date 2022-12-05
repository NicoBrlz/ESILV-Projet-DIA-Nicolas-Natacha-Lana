# -*- coding: utf-8 -*-

#Pour lancer depuis spyder/anaconda, effectuer la commande ci-dessous sur un terminal anaconda
#streamlit run "C:\Users\nberl\Desktop\Nicolas\ESILV\S7\Python\Projet\projet.py"

#%% Librairies
# essential libraries
import json
import random
import math
# storing and anaysis
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#modelisation
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#time
from datetime import datetime
from datetime import timezone

#API
import streamlit as st


#%% Dataframe
sc = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv", sep=',') 

def Nettoyer_dataframe(df): 

    #colonnes vides
    cols_vides = [col for col in df.columns if df[col].isnull().all()]
    df.drop(cols_vides, axis=1, inplace=True) #supprimer les colonnes vides
    
    #Enlever les lignes vides Nan
    df.dropna(axis =0, how='all', inplace=True)
    
    #Enlever les doublons
    df.drop_duplicates(keep="first", inplace = True)
Nettoyer_dataframe(sc)

def replace_question_mark(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x : np.NaN if(x == '?') else x)
replace_question_mark(sc)

sc.dropna(inplace = True)

from collections import defaultdict
col = ['Age','HoursPerWeek','TotalHours' ]
for i in range(len(col)):
    sc[col[i]] = sc[col[i]].astype(int)

#%% Analyses 

#Titre sur l'API
st.title('La league des joueurs de StarCraft II selon leurs statistiques')
st.subheader("Estimez votre rang à partir de vos statistiques grâce à une étude menée sur l'ensemble des joueurs!")
st.sidebar.title('Réglages') #Crée des réglages sur le côté

data = st.sidebar.checkbox('Afficher le dataframe')
if data:
    st.subheader('Dataset SkillCraft')
    st.dataframe(sc)

st.subheader('I. Les leagues')
st.sidebar.subheader('I. Les leagues')

#leagues

dict_league = {1: "Bronze", 2:"Silver", 3:"Gold", 4:"Platinum", 5:"Diamond", 6:"Master", 7:"GrandMaster", 8:"Professional"}
sc['Leagues'] = sc['LeagueIndex'].astype(str)
for i in range(1,9):
    sc['Leagues'] = sc['Leagues'].replace(to_replace=str(i),value=dict_league[i])
LeagueInd = sc.groupby('Leagues').count()
fig1 = px.histogram(LeagueInd, x=LeagueInd.index, y= "GameID" , category_orders= dict(Leagues = list(dict_league.values())) , title="Nombre de joueurs par Leagues")
fig1.update_layout(xaxis_title="Leagues" , yaxis_title="Nombre de joueurs")

league = st.sidebar.checkbox('Etudes sur les leagues')
if league: 

    

    opt = ["Actions par minute","Latence des actions","Nombre d'unités construites par unité de temps","Nombre de clics sur la mini carte par unité de temps", "Nombre de PAC par unité de temps"]
    opt2 = ['APM', 'ActionLatency', 'SelectByHotkeys', 'MinimapRightClicks', 'NumberOfPACs']
    txt = ["Le nombre d'actions par minute reste bien plus élevé dans la league GrandMaster que dans les autres."]
    txt.append("La colonnes ActionLatency semble un peu plus intéressante. Apparemment, la plupart des joueurs de la league GrandMaster ont une faible latence et réagissent donc rapidement aux événements (ce qui serait logique)")
    txt.append("Les joueurs des leagues Master et GrandMaster semblent sélectionner leurs unités à l'aide de touches de raccourci plus souvent que les joueurs des autres leagues.")
    txt.append("Ce graphe montre que plus un joueur clique sur sa minicarte, plus il est amené à avoir un niveau élevé. C'est à dire que plus un joueur sait prendre conscience de sa position et de celle des autres (on parle de 'Map awareness'), plus il sera capable de gagner. Cela ne semble cependant pas être un facteur déterminant.")
    txt.append("Le nombre de PAC (cycle Perception, Prédiction, Action, Résultat) est aussi très lié au niveau des joueurs. En effet, cette donnée correspond entre autre à la vitesse d'analyse de la situation d'un joueur, et donc à son adaptabilité.")
    option = st.sidebar.selectbox('Selectionnez une donnée', opt)
    fig2 = px.box(x=sc['LeagueIndex'],y=sc[opt2[opt.index(option)]], category_orders=list(dict_league.values()))
    fig2.update_layout(xaxis={'categoryorder' : 'array', 'categoryarray' : ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'GrandMaster', 'Professional'], 'title' : 'leagues'}, yaxis={'title' : option})

    st.write(fig1)
    st.write("Il y a plus de joueurs dans les leagues Platinum et Diamond. \nNous remarquons que la plupart des joueurs se situent dans les leagues 'moyennes'. Il semble qu'il soit facile de progresser vers la league Platinum/Diamond, mais pour atteindre la league Master ou GrandMaster, de sérieuses compétences sont nécessaires.")

    st.write('Etude sur les leagues')
    st.write(fig2)
    st.write(txt[opt.index(option)])

    colormap = sns.color_palette("YlGnBu")
    df = round(sc.corr(),2)
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(df, annot=True, cmap=colormap, fmt="")
    ax.set_title("Corrélation entre les variables du SkillCraft1")

    fig2, ax = plt.subplots()
    pd.plotting.scatter_matrix(sc[opt2], 
                            c=sc['LeagueIndex'],
                            cmap='YlGnBu',
                            figsize=(10, 10),
                            label=sc['LeagueIndex']);
    ax.set_title("Corrélation entre les stats et la league en nuage de points")



    corr = st.sidebar.checkbox("Afficher les corrélations avec le rang")
    if corr:
        st.write('Corrélation statistique/rang du joueur')
        st.pyplot(fig)
        st.pyplot(fig2)
    



#%% Modele

st.subheader('II. Le modèle')
st.sidebar.subheader('II. Le modèle')

X = sc.drop(['Age','GameID','TotalHours','TotalMapExplored','ComplexAbilitiesUsed','ComplexUnitsMade','UniqueUnitsMade','MinimapRightClicks','Leagues', 'LeagueIndex'], axis = 1)
Y = sc.LeagueIndex

X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scalert = preprocessing.StandardScaler().fit(X_test)
X_test = scalert.transform(X_test)


model = GradientBoostingClassifier(n_estimators = 80, max_depth = 3)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test[1].reshape(1, -1))

# %% Prédiction de résultats
mod = st.sidebar.checkbox('Prédire la leagues avec des statistiques')

if mod :
    st.sidebar.write()
    data1 = st.sidebar.number_input("Nombre d'heures par semaine (HoursPerWeek)", min_value=0.0, max_value=168.0)
    data2 = st.sidebar.number_input("Actions par minute (APM)", min_value=0.0, max_value=float(sc['APM'].max()))
    data3 = st.sidebar.number_input("Nombre de sélection d'unités ou de bâtiment avec un raccourci clavier (SelectByHotkeys)", min_value=0.0,max_value=float(sc['SelectByHotkeys'].max()))
    data4 = st.sidebar.number_input("Nombre de raccourcis clavier (AssignToHotkeys)", min_value=0.0, max_value=float(sc['AssignToHotkeys'].max()))
    data5 = st.sidebar.number_input("Raccourcis clavier uniques (UniqueHotkeys)", min_value=0.0, max_value=float(sc['UniqueHotkeys'].max()))
    data6 = st.sidebar.number_input("Attaques sur la minimap (MinimapAttacks)", min_value=0.0, max_value=float(sc['MinimapAttacks'].max()))
    data7 = st.sidebar.number_input("Nombre de cycles Perception/Attaque (NumberOfPACs)", min_value=0.0, max_value=float(sc['NumberOfPACs'].max()))
    data8 = st.sidebar.number_input("Intervalles des cycles Perception/Attaque (GapBetweenPACs)", min_value=0.0, max_value=float(sc['GapBetweenPACs'].max()))
    data9 = st.sidebar.number_input("Temps de réaction (ActionLatency)", min_value=0.0, max_value=float(sc['ActionLatency'].max()))
    data10 = st.sidebar.number_input("Actions durant les cycles Perception/Attaque (ActionsInPAC)", min_value=0.0, max_value=float(sc['ActionsInPAC'].max()))
    data11 = st.sidebar.number_input("Ouvriers générés (WorkersMade)", min_value=0.0, max_value=float(sc['WorkersMade'].max()))

    data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]
    df = pd.DataFrame(X.columns)

    test = st.sidebar.button('Prédire')
    if test :
        dict_image = {"Bronze" : "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d930a4ef-ff7f-46db-86ab-fdc00e874e22/d45ura4-0e4d9cc8-a76d-4300-8708-166143429f8a.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q5MzBhNGVmLWZmN2YtNDZkYi04NmFiLWZkYzAwZTg3NGUyMlwvZDQ1dXJhNC0wZTRkOWNjOC1hNzZkLTQzMDAtODcwOC0xNjYxNDM0MjlmOGEucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.S9y6fk1WxMUBX_SQWXtQ8SwjCmCtNa3rjzat8UfkqHQ",
                     "Silver" : "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d930a4ef-ff7f-46db-86ab-fdc00e874e22/d45us0s-3b0c0b21-dc76-4f6a-a219-91fdd9c39320.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q5MzBhNGVmLWZmN2YtNDZkYi04NmFiLWZkYzAwZTg3NGUyMlwvZDQ1dXMwcy0zYjBjMGIyMS1kYzc2LTRmNmEtYTIxOS05MWZkZDljMzkzMjAucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.NmrL-u0gJQsXx65G7LgQpqLlRR2gquSa-3r8dwPionU",
                     "Gold" : "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d930a4ef-ff7f-46db-86ab-fdc00e874e22/d45utzv-ba626858-bd36-4afb-a797-eb534bdbea05.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2Q5MzBhNGVmLWZmN2YtNDZkYi04NmFiLWZkYzAwZTg3NGUyMlwvZDQ1dXR6di1iYTYyNjg1OC1iZDM2LTRhZmItYTc5Ny1lYjUzNGJkYmVhMDUucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.13cEdl7joHu4UPxrRLuQcK5xlzsl0Wd8WFqDQ6Onn0Q",
                     "Platinum" : "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d930a4ef-ff7f-46db-86ab-fdc00e874e22/d45uqbc-1dc02a2c-781f-4924-a198-fde181d92580.png/v1/fill/w_900,h_1026,strp/platinum_league_icon_starcraft_by_corydbhs15_d45uqbc-fullview.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTAyNiIsInBhdGgiOiJcL2ZcL2Q5MzBhNGVmLWZmN2YtNDZkYi04NmFiLWZkYzAwZTg3NGUyMlwvZDQ1dXFiYy0xZGMwMmEyYy03ODFmLTQ5MjQtYTE5OC1mZGUxODFkOTI1ODAucG5nIiwid2lkdGgiOiI8PTkwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.QysRMWLimjfROpT9HRXEQQ7xzTJ9xtOQ5ROaM35xQwU",
                     "Diamond" : "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d930a4ef-ff7f-46db-86ab-fdc00e874e22/d464sdf-004cdad7-135c-44c4-b4e2-256ec8fb099f.png/v1/fill/w_900,h_1019,strp/diamond_league_icon_by_corydbhs15_d464sdf-fullview.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTAxOSIsInBhdGgiOiJcL2ZcL2Q5MzBhNGVmLWZmN2YtNDZkYi04NmFiLWZkYzAwZTg3NGUyMlwvZDQ2NHNkZi0wMDRjZGFkNy0xMzVjLTQ0YzQtYjRlMi0yNTZlYzhmYjA5OWYucG5nIiwid2lkdGgiOiI8PTkwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.pzhUvckoJxV33MD037stOYqRwBNQoje72dOxATTyFDU",
                     "Master" : "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/d930a4ef-ff7f-46db-86ab-fdc00e874e22/d47nbzv-b59850d7-0589-40e1-8b6b-25f4dc6c2dc1.png/v1/fill/w_900,h_1026,strp/masters_league_icon_by_corydbhs15_d47nbzv-fullview.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTAyNiIsInBhdGgiOiJcL2ZcL2Q5MzBhNGVmLWZmN2YtNDZkYi04NmFiLWZkYzAwZTg3NGUyMlwvZDQ3bmJ6di1iNTk4NTBkNy0wNTg5LTQwZTEtOGI2Yi0yNWY0ZGM2YzJkYzEucG5nIiwid2lkdGgiOiI8PTkwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.dAnURLWIxWeB5wNrFhDcbIVj4HE0tPy-aWh-_Geqxx8",
                     "GrandMaster" : "https://sc2ranks.net/images/badges/Grandmaster0.png"}
        pred = int(model.predict(scaler.transform(np.array(data).reshape(1, -1))))
        st.write('Selon vos statistiques, votre league est :')
        rank = dict_league[pred]

        st.image(dict_image[rank], width = 300)
        st.write(f"----> {rank} <----")
        
    

