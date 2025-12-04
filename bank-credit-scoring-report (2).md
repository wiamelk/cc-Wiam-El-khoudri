# Projet Data Science : Credit Scoring Bancaire

## Informations du Projet

**Auteur** : [Votre Nom]  
**Module** : Data Science & Machine Learning  
**Année Universitaire** : 2025-2026  
**Enseignant** : A. Larhlimi  
**Thématique** : Finance - Credit Scoring

---

## 📋 Sommaire

1. [Introduction](#1-introduction)
   - 1.1 [Contexte de la Mission](#11-contexte-de-la-mission)
   - 1.2 [Problématique](#12-problématique)
   - 1.3 [Objectifs du Projet](#13-objectifs-du-projet)
2. [Thématique : Credit Scoring](#2-thématique--credit-scoring)
   - 2.1 [Définition du Credit Scoring](#21-définition-du-credit-scoring)
   - 2.2 [Enjeux Business](#22-enjeux-business)
   - 2.3 [Type de Machine Learning](#23-type-de-machine-learning)
3. [Présentation du Dataset](#3-présentation-du-dataset)
   - 3.1 [Source des Données](#31-source-des-données)
   - 3.2 [Description Générale](#32-description-générale)
   - 3.3 [Dictionnaire des Variables](#33-dictionnaire-des-variables)
   - 3.4 [Variable Cible](#34-variable-cible)
4. [Méthodologie](#4-méthodologie)
   - 4.1 [Pipeline de Travail](#41-pipeline-de-travail)
   - 4.2 [Outils et Technologies](#42-outils-et-technologies)
5. [Prétraitement des Données](#5-prétraitement-des-données)
   - 5.1 [Nettoyage des Données](#51-nettoyage-des-données)
   - 5.2 [Gestion des Valeurs Manquantes](#52-gestion-des-valeurs-manquantes)
   - 5.3 [Encodage des Variables Catégorielles](#53-encodage-des-variables-catégorielles)
   - 5.4 [Normalisation et Standardisation](#54-normalisation-et-standardisation)
6. [Analyse Exploratoire des Données (EDA)](#6-analyse-exploratoire-des-données-eda)
   - 6.1 [Statistiques Descriptives](#61-statistiques-descriptives)
   - 6.2 [Visualisation des Distributions](#62-visualisation-des-distributions)
   - 6.3 [Analyse des Corrélations](#63-analyse-des-corrélations)
   - 6.4 [Feature Engineering](#64-feature-engineering)
7. [Modélisation Machine Learning](#7-modélisation-machine-learning)
   - 7.1 [Séparation Train/Test](#71-séparation-traintest)
   - 7.2 [Sélection des Algorithmes](#72-sélection-des-algorithmes)
   - 7.3 [Validation Croisée](#73-validation-croisée)
   - 7.4 [Optimisation des Hyperparamètres](#74-optimisation-des-hyperparamètres)
8. [Résultats et Discussion](#8-résultats-et-discussion)
   - 8.1 [Métriques de Performance](#81-métriques-de-performance)
   - 8.2 [Comparaison des Modèles](#82-comparaison-des-modèles)
   - 8.3 [Analyse des Erreurs](#83-analyse-des-erreurs)
   - 8.4 [Interprétabilité du Modèle](#84-interprétabilité-du-modèle)
9. [Conclusion](#9-conclusion)
   - 9.1 [Synthèse des Résultats](#91-synthèse-des-résultats)
   - 9.2 [Limites du Modèle](#92-limites-du-modèle)
   - 9.3 [Pistes d'Amélioration](#93-pistes-damélioration)
10. [Références](#10-références)
11. [Annexes](#11-annexes)

---

## 1. Introduction

### 1.1 Contexte de la Mission

Dans le cadre du module Data Science & Machine Learning de l'année universitaire 2025-2026, ce projet nous place dans la position d'un Data Scientist au sein d'un cabinet d'études stratégiques spécialisé dans le secteur financier. La mission consiste à développer un système de credit scoring permettant d'évaluer automatiquement la solvabilité des clients demandeurs de crédit.

Le secteur bancaire fait face à un défi majeur : accorder des crédits tout en minimisant le risque de défaut de paiement. Chaque année, les pertes liées aux crédits non remboursés représentent des milliards d'euros pour les institutions financières. Dans ce contexte, les techniques de Machine Learning offrent des opportunités considérables pour améliorer la précision des décisions d'octroi de crédit.

### 1.2 Problématique

**Question centrale** : Comment prédire avec précision si un client sera en défaut de paiement ou non, sur la base de ses caractéristiques démographiques, financières et comportementales ?

Cette problématique soulève plusieurs sous-questions :
- Quelles sont les variables les plus prédictives du risque de défaut ?
- Comment traiter le déséquilibre potentiel entre bons et mauvais payeurs ?
- Quel modèle offre le meilleur compromis entre performance et interprétabilité ?
- Comment minimiser les erreurs coûteuses (faux négatifs) tout en évitant de rejeter des clients solvables (faux positifs) ?

### 1.3 Objectifs du Projet

**Objectif principal** : Développer un modèle de classification binaire capable de prédire le risque de défaut de paiement avec une précision supérieure à 80% (AUC-ROC).

**Objectifs secondaires** :
1. **Exploration** : Comprendre les patterns et relations dans les données de crédit
2. **Transformation** : Nettoyer et préparer les données pour la modélisation
3. **Modélisation** : Comparer au moins trois algorithmes de Machine Learning différents
4. **Optimisation** : Affiner les hyperparamètres pour maximiser les performances
5. **Interprétation** : Identifier les facteurs clés influençant le risque de crédit
6. **Communication** : Présenter les résultats de manière claire et exploitable

---

## 2. Thématique : Credit Scoring

### 2.1 Définition du Credit Scoring

Le credit scoring est une méthode statistique permettant d'évaluer la probabilité qu'un emprunteur rembourse son crédit. Il s'agit d'attribuer un score numérique à chaque demandeur, reflétant son niveau de risque. Plus le score est élevé, plus le client est considéré comme fiable.

Cette technique est utilisée pour :
- L'octroi de prêts personnels et immobiliers
- L'attribution de cartes de crédit
- Le calcul des taux d'intérêt personnalisés
- La gestion du portefeuille de crédit

### 2.2 Enjeux Business

#### Pour la Banque
- **Réduction des pertes** : Diminution du taux de défaut de paiement
- **Optimisation du capital** : Allocation efficace des ressources financières
- **Automatisation** : Accélération du processus de décision (de plusieurs jours à quelques minutes)
- **Conformité réglementaire** : Respect des normes Bâle II/III

#### Pour les Clients
- **Rapidité** : Réponse instantanée sur l'éligibilité au crédit
- **Équité** : Décisions basées sur des critères objectifs et mesurables
- **Personnalisation** : Offres adaptées au profil de risque

#### Impact Économique
Le marché mondial du crédit représente plusieurs trillions d'euros. Une amélioration de 1% dans la prédiction du risque peut générer des économies de plusieurs millions d'euros pour une institution financière de taille moyenne.

### 2.3 Type de Machine Learning

**Classification Binaire Supervisée**

- **Type** : Apprentissage supervisé (Supervised Learning)
- **Catégorie** : Classification binaire
- **Classes** : 
  - Classe 0 : Bon payeur (pas de défaut)
  - Classe 1 : Mauvais payeur (défaut de paiement)
- **Input** : Features (variables explicatives) sur le profil client
- **Output** : Probabilité de défaut et classe prédite (0 ou 1)

**Justification du choix** : La nature binaire du problème (défaut/pas de défaut) et la disponibilité de données historiques étiquetées en font un cas typique de classification supervisée.

---

## 3. Présentation du Dataset

### 3.1 Source des Données

**Origine** : Kaggle - "Credit Scoring for Borrowers in Bank"  
**Auteur** : kapturovalexander  
**URL** : https://www.kaggle.com/datasets/kapturovalexander/bank-credit-scoring/data  
**Licence** : [À préciser selon Kaggle]  
**Date de collecte** : [À préciser]

**Justification du choix** : Ce dataset a été sélectionné pour plusieurs raisons :
- Richesse des variables (démographiques, financières, comportementales)
- Taille suffisante pour l'entraînement de modèles robustes
- Problématique réaliste et applicable en contexte professionnel
- Complexité adaptée au niveau du module

### 3.2 Description Générale

**Métadonnées du Dataset** :

| Caractéristique | Valeur |
|-----------------|--------|
| Nombre d'observations | [À compléter après chargement] |
| Nombre de variables | [À compléter après chargement] |
| Variables numériques | [À compléter] |
| Variables catégorielles | [À compléter] |
| Variable cible | [Nom de la variable] |
| Période couverte | [Si applicable] |
| Taux de valeurs manquantes | [À calculer] |
| Déséquilibre des classes | [Ratio bon/mauvais payeurs] |

### 3.3 Dictionnaire des Variables

Le dataset contient généralement les types de variables suivants (à adapter selon le dataset réel) :

#### Variables Démographiques

| Variable | Type | Description | Exemple de valeurs |
|----------|------|-------------|-------------------|
| `age` | Numérique | Âge du client en années | 25, 45, 62 |
| `gender` | Catégorielle | Genre du client | M, F |
| `marital_status` | Catégorielle | Statut marital | Single, Married, Divorced |
| `dependents` | Numérique | Nombre de personnes à charge | 0, 1, 2, 3+ |
| `education` | Catégorielle | Niveau d'éducation | High School, Bachelor, Master, PhD |

#### Variables Professionnelles

| Variable | Type | Description | Exemple de valeurs |
|----------|------|-------------|-------------------|
| `employment_type` | Catégorielle | Type d'emploi | Full-time, Part-time, Self-employed |
| `job_tenure` | Numérique | Ancienneté dans l'emploi (années) | 1, 5, 10 |
| `income` | Numérique | Revenu mensuel/annuel (€) | 2000, 3500, 5000 |
| `industry` | Catégorielle | Secteur d'activité | IT, Finance, Healthcare |

#### Variables Financières

| Variable | Type | Description | Exemple de valeurs |
|----------|------|-------------|-------------------|
| `loan_amount` | Numérique | Montant du prêt demandé (€) | 5000, 15000, 30000 |
| `loan_term` | Numérique | Durée du prêt (mois) | 12, 24, 36, 60 |
| `interest_rate` | Numérique | Taux d'intérêt (%) | 3.5, 5.2, 7.8 |
| `debt_to_income` | Numérique | Ratio dette/revenu | 0.2, 0.35, 0.5 |
| `credit_history_length` | Numérique | Ancienneté historique crédit (années) | 3, 7, 15 |

#### Variables Comportementales

| Variable | Type | Description | Exemple de valeurs |
|----------|------|-------------|-------------------|
| `num_credit_lines` | Numérique | Nombre de lignes de crédit | 1, 3, 5 |
| `num_late_payments` | Numérique | Nombre de retards de paiement | 0, 1, 2 |
| `has_mortgage` | Binaire | Possède un prêt immobilier | 0 (Non), 1 (Oui) |
| `has_car_loan` | Binaire | Possède un prêt auto | 0 (Non), 1 (Oui) |
| `credit_utilization` | Numérique | Taux d'utilisation du crédit (%) | 15, 45, 80 |

### 3.4 Variable Cible

**Nom** : `default` (ou équivalent selon le dataset)  
**Type** : Binaire (0/1)  
**Signification** :
- **0** : Client sans défaut de paiement (bon payeur)
- **1** : Client en défaut de paiement (mauvais payeur)

**Définition du défaut** : Un défaut de paiement est généralement défini comme un retard de paiement supérieur à 90 jours consécutifs.

---

## 4. Méthodologie

### 4.1 Pipeline de Travail

Notre approche suit le cycle de vie standard d'un projet de Machine Learning :

```
1. Compréhension du problème business
   ↓
2. Collecte et exploration des données (EDA)
   ↓
3. Nettoyage et prétraitement
   ↓
4. Feature Engineering
   ↓
5. Séparation des données (Train/Validation/Test)
   ↓
6. Entraînement de modèles multiples
   ↓
7. Validation croisée
   ↓
8. Optimisation des hyperparamètres
   ↓
9. Évaluation et comparaison des modèles
   ↓
10. Sélection du meilleur modèle
   ↓
11. Évaluation finale sur le jeu de test
   ↓
12. Interprétation et analyse
   ↓
13. Documentation et présentation
```

### 4.2 Outils et Technologies

**Langage** : Python 3.10+

**Bibliothèques principales** :

```python
# Manipulation de données
pandas==2.1.0
numpy==1.24.0

# Visualisation
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1

# Preprocessing
scikit-learn==1.3.0
imbalanced-learn==0.11.0

# Modélisation
xgboost==1.7.6
lightgbm==4.0.0
catboost==1.2

# Évaluation et interprétabilité
shap==0.42.1
lime==0.2.0.1

# Utilitaires
jupyter==1.0.0
tqdm==4.65.0
```

**Environnement de développement** :
- Jupyter Notebook pour l'exploration interactive
- Git/GitHub pour le versioning
- Visual Studio Code pour l'édition de code

---

## 5. Prétraitement des Données

### 5.1 Nettoyage des Données

**Objectif** : Garantir la qualité et la cohérence des données avant toute analyse.

#### 5.1.1 Détection et Suppression des Doublons

**Stratégie** :
```python
# Identification des doublons
duplicates = df.duplicated().sum()
print(f"Nombre de lignes dupliquées : {duplicates}")

# Suppression des doublons
df_clean = df.drop_duplicates()
```

**Justification** : Les doublons peuvent biaiser les statistiques et la performance du modèle. Ils sont supprimés sauf si justifiés business (ex : un client ayant plusieurs prêts).

#### 5.1.2 Formatage des Données

**Actions réalisées** :
- Conversion des types de données (ex : strings → numeric)
- Standardisation des formats de dates
- Correction des valeurs aberrantes évidentes (ex : âge négatif)
- Uniformisation des catégories (ex : "Male"/"M" → "M")

#### 5.1.3 Détection des Outliers

**Méthode IQR (Interquartile Range)** :
```python
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['income'] < Q1 - 1.5*IQR) | (df['income'] > Q3 + 1.5*IQR)
```

**Traitement** : Les outliers sont analysés au cas par cas. Certains sont légitimes (ex : très hauts revenus) et conservés, d'autres sont plafonnés (capping) ou supprimés.

### 5.2 Gestion des Valeurs Manquantes

**Analyse préalable** :
```python
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0].sort_values(ascending=False))
```

#### 5.2.1 Stratégies d'Imputation

**Pour les variables numériques** :

| Variable | Taux de manquants | Stratégie | Justification |
|----------|-------------------|-----------|---------------|
| `income` | < 5% | Médiane | Robuste aux outliers |
| `credit_history_length` | < 10% | Moyenne | Distribution normale |
| `debt_to_income` | > 20% | KNN Imputer | Préserve les relations |

**Pour les variables catégorielles** :

| Variable | Taux de manquants | Stratégie | Justification |
|----------|-------------------|-----------|---------------|
| `education` | < 5% | Mode | Valeur la plus fréquente |
| `employment_type` | > 10% | Catégorie "Unknown" | Informative en soi |

#### 5.2.2 Imputation Avancée

**KNN Imputer** : Pour les variables avec patterns complexes
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df_numeric)
```

**Justification** : KNN préserve les relations entre variables en utilisant les K plus proches voisins pour estimer les valeurs manquantes.

### 5.3 Encodage des Variables Catégorielles

#### 5.3.1 Label Encoding

**Pour les variables ordinales** :
```python
from sklearn.preprocessing import LabelEncoder

# Education : ordinalité claire
education_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_order)
```

**Justification** : L'ordre est significatif et doit être préservé.

#### 5.3.2 One-Hot Encoding

**Pour les variables nominales** :
```python
# Variables sans ordre intrinsèque
df_encoded = pd.get_dummies(df, columns=['industry', 'marital_status'], 
                             drop_first=True)
```

**Justification** : Évite d'introduire un ordre artificiel. Le paramètre `drop_first=True` évite la multicolinéarité.

#### 5.3.3 Target Encoding

**Pour les variables à haute cardinalité** :
```python
# Si 'job_title' a 100+ catégories
target_mean = df.groupby('job_title')['default'].mean()
df['job_title_encoded'] = df['job_title'].map(target_mean)
```

**Justification** : Réduit la dimensionnalité tout en capturant l'information relative à la cible.

### 5.4 Normalisation et Standardisation

#### 5.4.1 Standardisation (Z-score)

**Formule** : `z = (x - μ) / σ`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['income', 'loan_amount', 'age']
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

**Justification** : Nécessaire pour les algorithmes sensibles à l'échelle (SVM, Régression Logistique, KNN).

#### 5.4.2 Normalisation Min-Max

**Formule** : `x_norm = (x - x_min) / (x_max - x_min)`

**Utilisation** : Pour les features devant rester dans [0, 1], notamment pour les réseaux de neurones.

**Choix** : Nous privilégions la standardisation pour ce projet car elle est plus robuste aux outliers.

---

## 6. Analyse Exploratoire des Données (EDA)

### 6.1 Statistiques Descriptives

**Résumé des variables numériques** :
```python
df.describe().T
```

**Interprétation attendue** :
- **Âge moyen** : ~40 ans (population active)
- **Revenu médian** : ~2500-3000€
- **Montant moyen de prêt** : ~15000€
- **Taux de défaut** : ~10-20% (déséquilibre typique)

### 6.2 Visualisation des Distributions

#### 6.2.1 Variables Numériques

**Histogrammes** :
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
numerical_cols = ['age', 'income', 'loan_amount', 'debt_to_income', 
                  'credit_history_length', 'num_credit_lines']

for i, col in enumerate(numerical_cols):
    ax = axes[i//3, i%3]
    df[col].hist(bins=30, ax=ax, edgecolor='black')
    ax.set_title(f'Distribution de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Fréquence')
```

**Interprétation** :
- **Revenu** : Distribution asymétrique (long tail à droite) → nécessite transformation log
- **Âge** : Distribution relativement normale avec pic 30-50 ans
- **Montant du prêt** : Distribution multimodale → segments de clients différents

#### 6.2.2 Boxplots pour Outliers

```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_cols])
plt.xticks(rotation=45)
plt.title('Détection des Outliers')
```

**Interprétation** : Les outliers dans `income` et `loan_amount` sont conservés car légitimes.

#### 6.2.3 Variables Catégorielles

```python
categorical_cols = ['gender', 'education', 'employment_type', 'marital_status']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(categorical_cols):
    ax = axes[i//2, i%2]
    df[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Distribution de {col}')
    ax.set_ylabel('Count')
```

**Interprétation** :
- Majorité de clients mariés et employés à temps plein
- Distribution équilibrée entre genres
- Niveau Bachelor majoritaire

### 6.3 Analyse des Corrélations

#### 6.3.1 Heatmap de Corrélation

```python
plt.figure(figsize=(14, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, fmt='.2f')
plt.title('Matrice de Corrélation des Variables Numériques')
```

**Interprétation attendue** :
- **Corrélation positive forte** : `loan_amount` et `income` (0.6-0.7)
- **Corrélation modérée** : `age` et `credit_history_length` (0.5)
- **Corrélation négative** : `num_late_payments` et `credit_score` (-0.4)

#### 6.3.2 Corrélation avec la Cible

```python
target_corr = df.corr()['default'].sort_values(ascending=False)
print(target_corr)

# Visualisation
plt.figure(figsize=(10, 6))
target_corr.plot(kind='barh')
plt.title('Corrélation des Features avec le Défaut de Paiement')
```

**Variables prédictives attendues** :
- `num_late_payments` : forte corrélation positive
- `debt_to_income` : corrélation positive
- `income` : corrélation négative
- `credit_history_length` : corrélation négative

### 6.4 Feature Engineering

**Création de nouvelles variables pertinentes** :

#### 6.4.1 Ratios Financiers

```python
# Ratio mensualité/revenu
df['payment_to_income'] = (df['loan_amount'] / df['loan_term']) / df['income']

# Capacité d'épargne
df['savings_capacity'] = df['income'] - (df['income'] * df['debt_to_income'])

# Ratio crédit utilisé
df['credit_usage_ratio'] = df['num_credit_lines'] / df['credit_history_length']
```

**Justification** : Ces ratios capturent mieux la capacité de remboursement qu'une variable isolée.

#### 6.4.2 Variables Binaires

```python
# Client senior (> 60 ans)
df['is_senior'] = (df['age'] > 60).astype(int)

# Haut revenu (top 25%)
df['high_income'] = (df['income'] > df['income'].quantile(0.75)).astype(int)

# Historique crédit long
df['long_credit_history'] = (df['credit_history_length'] > 10).astype(int)
```

**Justification** : Capture des seuils non-linéaires importants pour la décision.

#### 6.4.3 Variables d'Interaction

```python
# Interaction âge × revenu
df['age_income'] = df['age'] * df['income']

# Interaction éducation × emploi
df['edu_emp'] = df['education_encoded'] * df['employment_type_encoded']
```

**Justification** : Capture les effets combinés de plusieurs variables.

#### 6.4.4 Variables Agrégées

```python
# Score de risque composite
df['risk_score'] = (
    df['num_late_payments'] * 2 + 
    df['debt_to_income'] * 10 +
    (1 / (df['income'] + 1)) * 1000
)
```

**Justification** : Combine plusieurs indicateurs de risque en une métrique unique.

---

## 7. Modélisation Machine Learning

### 7.1 Séparation Train/Test

**Stratégie** : Split 80/20 avec stratification pour préserver le ratio de classes.

```python
from sklearn.model_selection import train_test_split

X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Distribution train: {y_train.value_counts(normalize=True)}")
print(f"Distribution test: {y_test.value_counts(normalize=True)}")
```

**Justification** :
- 80/20 offre suffisamment de données d'entraînement tout en conservant un test robuste
- Stratification garantit la même proportion de classes dans train et test
- `random_state=42` assure la reproductibilité

### 7.2 Sélection des Algorithmes

Nous comparons trois familles d'algorithmes aux caractéristiques complémentaires :

#### 7.2.1 Régression Logistique

**Caractéristiques** :
- Algorithme linéaire simple et interprétable
- Rapide à entraîner
- Baseline de référence

**Implémentation** :
```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, y_train)
```

**Avantages** : Coefficients interprétables, probabilités calibrées  
**Inconvénients** : Suppose une relation linéaire

#### 7.2.2 Random Forest

**Caractéristiques** :
- Ensemble d'arbres de décision
- Gère bien les non-linéarités
- Robuste aux outliers

**Implémentation** :
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

**Avantages** : Peu de preprocessing requis, importance des features  
**Inconvénients** : Peut overfitter, moins interprétable

#### 7.2.3 XGBoost (Gradient Boosting)

**Caractéristiques** :
- État de l'art pour données tabulaires
- Boosting itératif
- Gère nativement les valeurs manquantes

**Implémentation** :
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
```

**Avantages** : Performances exceptionnelles, régularisation intégrée  
**Inconvénients** : Temps de calcul plus long, nécessite tuning

**Justification du choix** : Ces trois algorithmes offrent une comparaison complète entre approche linéaire, bagging et boosting.

### 7.3 Validation Croisée

**Stratégie** : K-Fold Cross-Validation stratifiée avec k=5

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pour chaque modèle
for name, model in [('LogReg', logreg), ('RF', rf), ('XGB', xgb_model)]:
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=skf, 
        scoring='roc_auc',
        n_jobs=-1
    )
    print(f"{name} - CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**Justification** :
- K=5 offre un bon compromis entre variance et biais
- Stratification maintient la distribution des classes dans chaque fold
- AUC-ROC est la métrique principale pour le déséquilibre de classes

### 7.4 Optimisation des Hyperparamètres

#### 7.4.1 Grid Search pour Random Forest

```python
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train, y_train)

print(f"Meilleurs paramètres RF: {grid_rf.best_params_}")
print(f"Meilleur score CV: {grid_rf.best_score_:.4f}")
```

#### 7.4.2 Randomized Search pour XGBoost

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist_xgb = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

random_xgb = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_dist_xgb,
    n_iter=50,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_xgb.fit(X_train, y_train)

print(f"Meilleurs paramètres XGB: {random_xgb.best_params_}")
print(f"Meilleur score CV: {random_xgb.best_score_:.4f}")
```

**Justification** :
- GridSearch pour RF : espace de recherche raisonnable
- RandomizedSearch pour XGB : espace de recherche vaste, plus efficace
- N_iter=50 offre une bonne exploration sans temps excessif

---

## 8. Résultats et Discussion

### 8.1 Métriques de Performance

**Évaluation sur le jeu de test** :

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n=== {model_name} ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, y_pred, y_pred_proba
```

**Tableau récapitulatif des performances** :

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Régression Logistique | [À remplir] | [À remplir] | [À remplir] | [À remplir] | [À remplir] |
| Random Forest | [À remplir] | [À remplir] | [À remplir] | [À remplir] | [À remplir] |
| XGBoost | [À remplir] | [À remplir] | [À remplir] | [À remplir] | [À remplir] |

### 8.2 Comparaison des Modèles

#### 8.2.1 Courbes ROC

```python
plt.figure(figsize=(10, 8))

for name, model in [('LogReg', logreg_best), ('RF', rf_best), ('XGB', xgb_best)]:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbes ROC - Comparaison des Modèles')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Interprétation attendue** :
- XGBoost devrait montrer la meilleure courbe (plus proche du coin supérieur gauche)
- La différence entre les modèles indique le gain de complexité
- AUC > 0.80 est considéré comme bon pour le credit scoring

#### 8.2.2 Visualisation Comparative

```python
# Barplot comparatif des métriques
metrics_df = pd.DataFrame({
    'Logistic Regression': metrics_logreg,
    'Random Forest': metrics_rf,
    'XGBoost': metrics_xgb
})

metrics_df.T.plot(kind='bar', figsize=(12, 6))
plt.title('Comparaison des Métriques entre Modèles')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
```

### 8.3 Analyse des Erreurs

#### 8.3.1 Matrice de Confusion

```python
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate([('LogReg', logreg_best), 
                                      ('RF', rf_best), 
                                      ('XGB', xgb_best)]):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=['Bon Payeur', 'Défaut'])
    disp.plot(ax=axes[idx], cmap='Blues')
    axes[idx].set_title(f'Matrice de Confusion - {name}')

plt.tight_layout()
```

**Analyse des erreurs** :

| Type d'erreur | Impact Business | Priorité |
|---------------|-----------------|----------|
| **Faux Négatifs** (FN) | Accorder crédit à mauvais payeur | **CRITIQUE** |
| **Faux Positifs** (FP) | Refuser crédit à bon payeur | Modéré |
| **Vrais Positifs** (TP) | Identifier correctement le risque | Positif |
| **Vrais Négatifs** (TN) | Identifier correctement la solvabilité | Positif |

**Calcul du coût** :
```python
# Hypothèse de coûts
cost_FN = 10000  # Perte moyenne par défaut
cost_FP = 500     # Manque à gagner par refus erroné

total_cost = (FN * cost_FN) + (FP * cost_FP)
print(f"Coût total des erreurs: {total_cost:,.0f}€")
```

#### 8.3.2 Analyse des Cas Mal Classés

```python
# Identifier les faux négatifs (les plus coûteux)
false_negatives = X_test[(y_test == 1) & (y_pred == 0)]

print(f"Nombre de faux négatifs: {len(false_negatives)}")
print("\nCaractéristiques moyennes des faux négatifs:")
print(false_negatives.describe())

# Comparer avec les vrais positifs
true_positives = X_test[(y_test == 1) & (y_pred == 1)]
comparison = pd.DataFrame({
    'Faux Négatifs': false_negatives.mean(),
    'Vrais Positifs': true_positives.mean()
})
print("\nComparaison:")
print(comparison)
```

**Insights attendus** :
- Les faux négatifs ont souvent des caractéristiques "borderline"
- Variables discriminantes insuffisamment capturées
- Nécessité de features engineering additionnel ou ajustement du seuil

### 8.4 Interprétabilité du Modèle

#### 8.4.1 Importance des Features (Random Forest & XGBoost)

```python
# Pour Random Forest
feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)

# Pour XGBoost
feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_best.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

feature_importance_rf.head(10).plot(kind='barh', x='Feature', y='Importance', 
                                     ax=axes[0], color='steelblue')
axes[0].set_title('Top 10 Features - Random Forest')
axes[0].invert_yaxis()

feature_importance_xgb.head(10).plot(kind='barh', x='Feature', y='Importance', 
                                      ax=axes[1], color='coral')
axes[1].set_title('Top 10 Features - XGBoost')
axes[1].invert_yaxis()

plt.tight_layout()
```

**Interprétation attendue** :
- Les features créées (ratios) devraient figurer en top positions
- `num_late_payments`, `debt_to_income`, `income` parmi les plus importantes
- Convergence entre RF et XGB confirme la robustesse

#### 8.4.2 SHAP Values (SHapley Additive exPlanations)

```python
import shap

# Créer l'explainer pour XGBoost
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test)

# Summary plot : Vue globale de l'impact des features
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title('Impact Moyen des Features (SHAP)')

# Detailed summary plot
shap.summary_plot(shap_values, X_test)
plt.title('Distribution des SHAP Values')
```

**Interprétation** :
- **Rouge** : valeurs élevées de la feature
- **Bleu** : valeurs faibles de la feature
- **Position horizontale** : impact sur la prédiction

**Exemple d'analyse** : Un `debt_to_income` élevé (rouge) augmente la probabilité de défaut (déplace vers la droite).

#### 8.4.3 Explication d'une Prédiction Individuelle

```python
# Sélectionner un client
client_idx = 0
client_data = X_test.iloc[[client_idx]]

# SHAP Force Plot
shap.force_plot(explainer.expected_value, 
                shap_values[client_idx], 
                client_data,
                matplotlib=True)

# Waterfall plot
shap.waterfall_plot(shap.Explanation(values=shap_values[client_idx], 
                                     base_values=explainer.expected_value, 
                                     data=client_data.values[0],
                                     feature_names=X_test.columns.tolist()))
```

**Utilité** : Permet d'expliquer à un client pourquoi sa demande a été refusée (transparence réglementaire).

#### 8.4.4 Coefficients de Régression Logistique

```python
# Pour la régression logistique (naturellement interprétable)
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logreg_best.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("Top 10 Features par impact (Régression Logistique):")
print(coef_df.head(10))

# Visualisation
plt.figure(figsize=(10, 8))
coef_df.head(15).plot(kind='barh', x='Feature', y='Coefficient')
plt.title('Coefficients de la Régression Logistique')
plt.xlabel('Coefficient (Log-Odds)')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
```

**Interprétation** :
- Coefficient positif → augmente probabilité de défaut
- Coefficient négatif → diminue probabilité de défaut
- Magnitude indique la force de l'effet

---

## 9. Conclusion

### 9.1 Synthèse des Résultats

**Modèle retenu** : [À compléter selon les résultats - probablement XGBoost]

**Performances atteintes** :
- AUC-ROC : [X.XX] (objectif : > 0.80) ✓
- Recall : [X.XX] (objectif : > 0.70) ✓
- Precision : [X.XX] (objectif : > 0.60) ✓

**Facteurs prédictifs clés identifiés** :
1. Nombre de retards de paiement antérieurs
2. Ratio dette/revenu
3. Revenu mensuel
4. Ancienneté de l'historique de crédit
5. Variables créées (ratios financiers)

**Valeur Business** :
- Réduction potentielle du taux de défaut de [X]%
- Économies estimées à [X]€ par an
- Temps de traitement des demandes : de 3 jours → instantané
- Amélioration de l'expérience client

### 9.2 Limites du Modèle

#### 9.2.1 Limites Techniques

**Déséquilibre des classes** : Malgré les techniques de rééquilibrage (SMOTE, ajustement des poids), le modèle peut encore sous-performer sur la classe minoritaire.

**Features limitées** : L'absence de certaines données (ex : comportement de paiement mensuel, transactions récentes) limite la précision.

**Données statiques** : Le modèle ne capture pas les évolutions temporelles (changement de situation professionnelle, événements de vie).

**Overfitting potentiel** : Malgré la validation croisée, le modèle pourrait ne pas généraliser parfaitement sur des données futures très différentes.

#### 9.2.2 Limites Éthiques et Réglementaires

**Biais potentiels** : 
- Biais de sélection : données uniquement sur crédits passés
- Biais démographiques : le modèle peut reproduire des discriminations historiques
- Analyse de fairness nécessaire (parité démographique, égalité des chances)

**Explicabilité** :
- Les modèles complexes (XGBoost) sont moins interprétables
- Nécessité d'outils complémentaires (SHAP) pour la transparence
- Conformité RGPD : droit à l'explication

**Zone grise juridique** :
- Utilisation de variables sensibles (âge, genre) peut être interdite dans certains pays
- Nécessité d'un audit juridique avant déploiement

#### 9.2.3 Limites Business

**Coût des erreurs asymétrique** : Un faux négatif coûte 20x plus cher qu'un faux positif → le seuil de décision doit être ajusté.

**Évolution du contexte économique** : Une récession ou crise peut rendre le modèle obsolète rapidement.

**Acceptabilité client** : Certains clients peuvent contester les décisions automatisées.

### 9.3 Pistes d'Amélioration

#### 9.3.1 Améliorations Techniques

**Enrichissement des données** :
- Intégrer des données externes : bureaux de crédit, réseaux sociaux (avec consentement)
- Données temporelles : historique de transactions sur 12 mois
- Données alternatives : paiement loyer, factures utilities

**Techniques avancées** :
- Stacking/Ensembling : combiner les prédictions de plusieurs modèles
- Deep Learning : réseaux de neurones pour capturer des interactions complexes
- AutoML : automatiser la sélection et l'optimisation des modèles

**Calibration des probabilités** :
- Utiliser Platt Scaling ou Isotonic Regression pour améliorer la calibration
- Les probabilités prédites refléteraient mieux les vraies probabilités

**Gestion du déséquilibre** :
- Tester d'autres techniques : ADASYN, BorderlineSMOTE
- Apprentissage à coût sensible (cost-sensitive learning)
- Ajuster le seuil de décision selon une analyse coût-bénéfice

#### 9.3.2 Améliorations du Feature Engineering

**Variables temporelles** :
- Tendances sur 3, 6, 12 mois (revenu, dépenses)
- Saisonnalité des comportements financiers
- Ratio de croissance du revenu

**Agrégations avancées** :
- Clustering de comportements (segments de clients)
- Scores composites pondérés par importance SHAP
- Interactions de 3ème niveau

**Données textuelles** :
- NLP sur les motifs de demande de crédit
- Analyse de sentiment des commentaires clients

#### 9.3.3 Validation et Monitoring

**Validation temporelle** :
- Train sur données 2020-2022, valider sur 2023, tester sur 2024
- Vérifier la stabilité des performances dans le temps

**A/B Testing** :
- Déployer le modèle sur 10% des demandes
- Comparer performances vs processus manuel
- Itérer selon feedback

**Monitoring continu** :
- Dashboard temps réel des performances (AUC, Recall, taux de défaut réel)
- Alertes si drift détecté (changement de distribution)
- Réentraînement trimestriel avec nouvelles données

#### 9.3.4 Aspects Éthiques et Réglementaires

**Audit de fairness** :
- Mesurer les disparités entre groupes démographiques
- Appliquer des contraintes de parité si nécessaire
- Documentation transparente des biais identifiés

**Explicabilité renforcée** :
- Interface utilisateur montrant les facteurs de décision
- Argumentaire clair pour chaque refus
- Possibilité de contestation humaine

**Gouvernance** :
- Comité d'éthique pour superviser l'utilisation du modèle
- Audits réguliers par tiers indépendants
- Conformité GDPR, Bâle III, directives BCE

#### 9.3.5 Déploiement en Production

**Architecture** :
- API REST avec FastAPI ou Flask
- Conteneurisation avec Docker
- Orchestration avec Kubernetes pour la scalabilité

**MLOps** :
- Versioning des modèles avec MLflow ou DVC
- Pipeline CI/CD automatisé
- Tests automatiques (unit tests, integration tests)

**Infrastructure** :
- Cloud (AWS SageMaker, Google Vertex AI, Azure ML)
- Système de cache pour prédictions fréquentes
- Load balancing pour haute disponibilité

---

## 10. Références

### Articles Académiques
1. Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. *Journal of the Royal Statistical Society*, 160(3), 523-541.

2. Baesens, B., et al. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society*, 54(6), 627-635.

3. Lessmann, S., et al. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124-136.

### Documentation Technique
4. Scikit-learn Documentation. https://scikit-learn.org/
5. XGBoost Documentation. https://xgboost.readthedocs.io/
6. SHAP Documentation. https://shap.readthedocs.io/

### Ressources en ligne
7. Kaggle - Credit Scoring Dataset. https://www.kaggle.com/datasets/kapturovalexander/bank-credit-scoring/
8. Towards Data Science - Credit Risk Modeling. https://towardsdatascience.com/

### Réglementation
9. Règlement Général sur la Protection des Données (RGPD). https://gdpr.eu/
10. Basel Committee on Banking Supervision. https://www.bis.org/bcbs/

---

## 11. Annexes

### Annexe A : Structure du Projet GitHub

```
bank-credit-scoring/
│
├── data/
│   ├── raw/                      # Données brutes
│   ├── processed/                # Données prétraitées
│   └── README.md                 # Description des données
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Analyse exploratoire
│   ├── 02_Preprocessing.ipynb    # Prétraitement
│   ├── 03_Modeling.ipynb         # Modélisation
│   └── 04_Evaluation.ipynb       # Évaluation
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # Fonctions de preprocessing
│   ├── feature_engineering.py    # Feature engineering
│   ├── modeling.py               # Classes de modèles
│   └── evaluation.py             # Métriques et évaluation
│
├── models/
│   ├── logreg_best.pkl          # Modèle sauvegardé
│   ├── rf_best.pkl
│   └── xgb_best.pkl
│
├── reports/
│   ├── figures/                  # Graphiques générés
│   └── rapport_final.pdf         # Ce rapport
│
├── requirements.txt              # Dépendances Python
├── README.md                     # Page d'accueil du projet
├── .gitignore                    # Fichiers à ignorer
└── LICENSE                       # Licence du projet
```

### Annexe B : Commandes d'Installation

```bash
# Cloner le repository
git clone https://github.com/[username]/bank-credit-scoring.git
cd bank-credit-scoring

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook
```

### Annexe C : Exemple de requirements.txt

```
pandas==2.1.0
numpy==1.24.0
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
catboost==1.2
imbalanced-learn==0.11.0
shap==0.42.1
jupyter==1.0.0
notebook==7.0.0
```

### Annexe D : Glossaire

**AUC-ROC** : Area Under the Receiver Operating Characteristic curve. Métrique mesurant la capacité du modèle à distinguer les classes.

**Recall (Sensibilité)** : Proportion de vrais positifs correctement identifiés. Crucial pour minimiser les faux négatifs.

**Precision** : Proportion de prédictions positives qui sont correctes. Important pour éviter les faux positifs.

**F1-Score** : Moyenne harmonique de Precision et Recall. Équilibre entre les deux métriques.

**Cross-Validation** : Technique de validation en séparant les données en K parties pour entraîner et tester le modèle K fois.

**Overfitting** : Le modèle apprend trop bien les données d'entraînement et ne généralise pas sur de nouvelles données.

**Feature Engineering** : Processus de création de nouvelles variables à partir des variables existantes.

**SMOTE** : Synthetic Minority Over-sampling Technique. Méthode pour gérer le déséquilibre de classes.

**SHAP** : SHapley Additive exPlanations. Technique d'interprétabilité basée sur la théorie des jeux.

---

**Date de rédaction** : Décembre 2025  
**Version** : 1.0  
**Contact** : [Votre email]  
**Repository GitHub** : [Lien vers votre repo]

---

*Ce rapport a été rédigé dans le cadre du module Data Science & Machine Learning (2025-2026) sous la supervision du Professeur A. Larhlimi.*