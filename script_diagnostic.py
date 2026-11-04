# 1- Création du dataframe
# 1.1- Importation des bibliothèques
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1.2- Chargement du jeu de données
data = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
print(data.head())

# 1.3- Nettoyage des noms des colonnes
data.columns

# 1.3.1- Mettre en minuscules
data.columns = data.columns.str.lower()
data.columns

# 1.3.2- Remplacer les espaces par "_"
data.columns = data.columns.str.replace(' ', '_')
data.columns

# 1.3.3- Remplacer "-" par "_"
data.columns = data.columns.str.replace('-', '_')
data.columns

# 1.3.4- Renommer les colonnes "lat" et "lon" par "latitude" et "longitude"
data = data.rename(columns={'lat':'latitude'})
data = data.rename(columns={'lon':'longitude'})
data.columns

# 1.4- Types des variables
print("Lignes au départ :", len(data))

# 1.5 Taux de complétude par variable (%)
completeness = (1 - data.isna().mean()) * 100
for c in data.columns:
    if data[c].dtype.name in ('category','object'):
        pass
print(completeness.sort_values().to_string())

# 2. Diagnostic du jeu de données
print("\nValeurs manquantes (%) :")
print((data.isna().mean() * 100).round(1).sort_values(ascending=False).head(10))

# 2.1- Doublons (identiques sur tout sauf dr_no)
avant = len(data)
data = data.drop_duplicates(subset=[c for c in data.columns if c != 'dr_no'])
print("\nDoublons supprimés :", avant - len(data))

# 2.2- Colonnes inutiles (vides ou redondantes ou non pertinentes dans l'analyse)
colonnes_inutiles = ['latitude', 'longitude', 'location', 'rpt_dist_no',
                     'mocodes', 'weapon_used_cd', 'vict_descent', 'date_rptd',
                     'crm_cd_1', 'crm_cd_2', 'crm_cd_3', 'crm_cd_4',
                     'cross_street', 'area', 'status', 'premis_cd']
data = data.drop(columns=inutiles)

# crm_cd_4 :        100 % vide (aucune information)
# crm_cd_3 :        99,8 % vide (aucune information)
# crm_cd_2 :        93 % vide (crime secondaire rarissime, trop peu de cas pour une analyse)
# cross_street :    84,5 % vide (rue transversale, seulement pour certaines adresses)
# crm_cd_1 :        identique à Crm Cd dans 99,8 % des cas (doublon de variable)
# area :            code numérique de area_name (même information, le libellé est plus lisible)
# status :          code de status_desc (même information, le libellé suffit)
# premis_cd :       code de premis_desc (même information, le libellé suffit)
# weapon_sed_cd est gardée un instant seulement (elle sert à repérer les armes à feu, puis elle est retirée, car weapon_desc contient la même information).

# 2.3- Valeurs aberrantes -> manquantes (pas d'imputation)
data.loc[data['vict_age'] <= 0, 'vict_age'] = np.nan  # âge 0 ou négatif
data.loc[data['vict_age'] > 100, 'vict_age'] = np.nan
data.loc[(data['latitude'] == 0) & (data['longitude'] == 0),
['latitude', 'longitude']] = np.nan

# 2.4- Incohérences : codes hors référentiel
data.loc[~data['vict_sex'].isin(['M', 'F', 'X']), 'vict_sex'] = np.nan
data.loc[data['vict_descent'] == '-', 'vict_descent'] = np.nan

# 2.5- Valeurs manquantes informatives -> libellés explicites
data['weapon_desc'] = data['weapon_desc'].fillna('Aucune arme déclarée')
data['premis_desc'] = data['premis_desc'].fillna('Inconnu')
data['vict_sex'] = data['vict_sex'].fillna('Non applicable')
data['vict_descent'] = data['vict_descent'].fillna('Non applicable')

# 2.7- Nouvelles variables (utiles)
data['date_occ'] = pd.to_datetime(data['date_occ'],
                                  format='%m/%d/%Y %I:%M:%S %p')
data['annee'] = data['date_occ'].dt.year
data['mois'] = data['date_occ'].dt.month
data['jour_semaine'] = data['date_occ'].dt.day_name()
# heure : time_occ est du texte « 2130 » ; l'heure entière (0-23) permet l'analyse horaire.
data['heure'] = data['time_occ'].str.zfill(4).str[:2].astype(int)
# heure_midi_exact : 12h00 pile est une heure par défaut (excès de ~19 600 cas) ; l'indicateur permet
# de corriger le faux pic de midi au lieu de conclure à tort que le crime culmine à 12h.
data['heure_midi_exact'] = (data['time_occ'].str.zfill(4) == '1200')  # heure par défaut
# arme_utilisee : remplace weapon_used_cd (vide à 67 %) par un simple oui/non, directement
data['arme_utilisee'] = data['weapon_desc'] != 'Aucune arme déclarée'
# plage : 24 heures sont trop détaillées pour une lecture opérationnelle ; 4 plages répondent à
# « quelle période de la journée est la plus touchée ? » (à comparer par heure de plage, durées inégales).
data['plage'] = pd.cut(data['heure'], [-1, 4, 11, 16, 20, 23],
                     labels=['nuit', 'matin', 'après-midi', 'soirée', 'nuit'], ordered=False)

data.to_csv('Crime_Data_Nettoye_Simple.csv', index=False, encoding='utf-8-sig')
print("\nLignes après nettoyage :", len(data), "| colonnes :", data.shape[1])

##########################################################################################
# 3- Transformation simplifiée : enrichit le fichier nettoyé.
data = pd.read_csv(sys.argv[1], parse_dates=['date_occ])

# 3.1- Armes : codes LAPD (101-110 et 115-125 = armes à feu ; 111-114 = imitations)
# arme_a_feu : le critère « arme présente » mélange force physique et menaces verbales (plus de la
# moitié des cas). Les codes LAPD (101-110 et 115-125 = armes à feu ; 111-114 = imitations) isolent les
# vraies armes à feu, ce que la recherche par mots-clés faisait mal (jouets comptés, revolvers oubliés).
code = pd.to_numeric(data['weapon_used_cd'], errors='coerce')
data['arme_a_feu'] = code.between(101, 110) | code.between(115, 125)

# 3.2- Week-end
# weekend : teste directement l'hypothèse « le week-end est-il plus touché ? » en un seul indicateur.
data['weekend'] = data['jour_semaine'].isin(['saturday', 'sunday'])

# 3.3- Score de gravité = 2 (crime Partie 1) + 1 (arme) + 2 (arme à feu)
# gravite_score : le jeu de données n'a aucune variable de gravité, or le projet cherche les facteurs
data['gravite_score'] = ((data['part_1_2'] == 1) * 2) + data['arme_utilisee'] * 1 + data['arme_a_feu'] * 2
# Gravite: 4 niveaux lisibles(Faible, Modérée, Élevée, Très élevée) pour
les tableaux, tests et modèles.
data['gravite'] = data['gravite_score'].map(lambda s: 'Faible' if s == 0 else 'Modérée' if s <= 2
                                        else 'Élevée' if s == 3 else 'Très élevée')

# 3.4- Encodage numérique des variables texte (pour les modèles)
# Encodages *_code : les modèles (forêts aléatoires) n'acceptent que des nombres ; zone, type de
# lieu, sexe et type de crime sont donc convertis en codes numériques (les libellés sont conservés).
for c in ['area_name', 'premis_desc', 'vict_sex', 'crm_cd_desc']:
    data[c + '_code'] = data[c].astype('category').cat.codes

# 3.5- Ne garder que les colonnes pertinentes pour l'analyse et les modèles
colonnes_gardees = ['dr_nO', 'date_occ', 'annee', 'mois', 'jour_semaine', 'heure', 'heure_midi_exact', 'plage', 'weekend',
          'area_name', 'premis_desc', 'crm_cd_desc', 'part_1_2', 'vict_age', 'vict_sex',
          'weapon_desc', 'arme_utilisee', 'arme_a_feu', 'status_desc', 'gravite_score', 'gravite',
          'area_name_code', 'premis_desc_code', 'vict_sex_code', 'crm_cd_desc_code']
data = data[colonnes_gardees]

# Colonnes retirées car non utilisées dans l'analyse ni dans les modèles :
# latitude et longitude :   l'analyse géographique se fait par zone de police (AREA NAME, 21 zones) ;
                            # les coordonnées ne servent qu'à une carte fine, hors du périmètre actuel
# location :                adresse en texte, quasi unique à chaque ligne, redondante avec latitude/longitude
# rpt_ist_no :              1 209 districts, trop fin pour les modèles ; la zone suffit
# mocodes :                 codes de mode opératoire multiples, 15 % vides, trop complexes à exploiter ici
# weapon_used_cd :          déjà résumée par Weapon Desc et les indicateurs d'arme (utilisée plus haut)
# vict_descent :            non utilisée dans les analyses ni les modèles de ce projet
# crm_cd :                  code de Crm Cd Desc, même information
# date_rptd :               seule la date de survenue sert à l'analyse des tendances

data.to_csv('Crime_Data_Enrichi_Simple.csv', index=False, encoding='utf-8-sig')
print(data['gravite'].value_counts(normalize=True).round(3))
print("Colonnes :", data.shape[1])
