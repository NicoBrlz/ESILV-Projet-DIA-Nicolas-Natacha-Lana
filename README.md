# ESILV-Projet-DIA-Nicolas-Natacha-Lana
Répertoire du projet DIA 2022

Nicolas BERLIOZ, Lana BONHOMME, Natacha Batmini BATMANABANE
Groupe DIA1

# Contenu
- le dataset (.csv)
- le notebook (.ipynb)
- le code pour l'API streamlit (.py)
- la présentation (.pdf)

# Le dataset
Le dataset contient des données concernant les statistiques et la league de joueurs de Starcraft II, un jeu en ligne axé sur le mode compétitif, ayant une scène e-sport d'envergure mondiale.
Lien : http://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv

# Nos analyses
lien du notebook (vous le trouverez aussi sur le github) :
https://colab.research.google.com/drive/1RphVNvVR_t_WZ6qGyxUMtzR1Koo-ZH42?usp=sharing&fbclid=IwAR2wb5I6nz5NoJiGgbMVQ_hyt6RFy232asWHyBYNcY08mcFfVGTBi-j3yPY#scrollTo=TYRspZfosIos


Le meilleur modèle est un Random Forest Classifier avec une précision (accuracy) de 0,63. C'est celui qu'on utilse dans notre API pour déterminer la league des joueurs en rentrant leurs statistiques dans les réglages.

Nous voulions savoir comment déterminer la league d'un joueur à partir de ces statistiques.

Les caractéristiques pour déterminer la leagued'un joueur sont 'HoursPerWeek', 'Actions par minute', 'SelectByHotkeys', 'AssignToHotkeys','UniqueHotkeys', 'MinimapAttacks', 'NumberOfPACs', 'GapBetweenPACs','ActionLatency', 'ActionsInPAC', 'WorkersMade'.
Nous les avons choisies en prenant les variables ayant une bonne corrélation avec la League, puis en supprimant certaines ayant une trop forte corrélation entre elles afin d'éviter l'overfitting.

Problème rencontrés:

Il nous manque des informations sur la League 8, donc nous avons pris la décision de supprimer les lignes des joeurs de League 8 pour la partie prédiction. La prédiction de leur league est simple étant donné qu'il y avait deux valeurs NaN sur les colonnes 'Age' et 'HoursPerWeek' et que ce sont les seules lignes ayant ces particularités.
Nous nous sommes retrouvés avec des résultats en dessous de nos attentes. Nous avions commencé par de la classification sur toutes les colonnes de notre dataset et nous arrivions à des résultats d'accuracy de 0,3. Nous avons donc essayé de faire de la régression qui ne nous a pas donné de meilleurs résultats.
Nous nous sommes alors concentrés sur de la classification en selectionnant les colonnes ('HoursPerWeek', 'Actions par minute', 'SelectByHotkeys', 'AssignToHotkeys','UniqueHotkeys', 'MinimapAttacks', 'NumberOfPACs', 'GapBetweenPACs','ActionLatency', 'ActionsInPAC', 'WorkersMade').
Nous avons également essayé de prédire si le joeur était dans une league qui lui correspondait. Notre modèle est limité car à partir de la League 7 il ne fonctionne plus comme précisé dans notre code. C'est pour cela que nous avons abandonné ce modèle.
Meilleur résultat obtenu Nous avons obtenu notre meilleur résultat en faisant de la classification sur les colonnes citées précédement à l'aide d'un GradientBostingClassifier.


# Outils utilisés
- Python
- Jupyter Notebook
- Python Packages:
	- `numpy`
	- `pandas`
	- `matplotlib`
	- `sklearn`
	- `statsmodels`
	- `seaborn`
	- `streamlit`
- Présentation des analyses en pdf
