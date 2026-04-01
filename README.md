# Projet Big Data End-to-End : traitement distribué de 541 909 transactions e-commerce avec PySpark, segmentation client non supervisée (RFM + GaussianMixture, Silhouette = 0.608) et prédiction des "gros dépensiers" via RandomForest (AUC = 0.918).

🎯 Contexte & Problématique

Ce projet s'appuie sur un jeu de données de type Online Retail regroupant des transactions clients d'une plateforme e-commerce internationale. Deux objectifs business guident l'analyse :

Segmenter la clientèle en groupes homogènes à partir de variables RFM pour personnaliser les actions marketing (approche non supervisée).
Prédire les gros dépensiers (Monetary > 2 000 €) afin d'optimiser le ciblage commercial (approche supervisée).


🏗️ Architecture du Pipeline

[Données brutes]           [Prétraitement]          [Modélisation]            [Restitution]
Online Retail CSV    →    Nettoyage PySpark   →    RFM + GMM             →   Visualisations
541 909 transactions       Filtrage anomalies        (Clustering)              Matplotlib /
InvoiceNo                  Cast types                +                         Seaborn
StockCode                  Feature Engineering       RandomForest              Recommandations
Quantity                   VectorAssembler           (Classification)          métier
UnitPrice                  StandardScaler
CustomerID                 Agrégation RFM
InvoiceDate
Country

🗂️ Structure du Dépôt

ecommerce-pyspark-bigdata/
├── notebook/         → déposez votre .ipynb ici
├── data/             → créez le dossier (vide, le .gitignore exclut les CSV)
├── slides/           → déposez vos slides ici
├── outputs/          → pour les exports CSV et métriques
├── README.md         ← fichier fourni
├── requirements.txt  ← fichier fourni
└── .gitignore        ← fichier fourni

📋 Données

<img width="392" height="280" alt="image" src="https://github.com/user-attachments/assets/91b99656-f4fb-4695-bf56-db7a20c845d1" />

📥 Disponible sur Kaggle — Online Retail Dataset.


🔬 Étapes du Projet

1️⃣ Initialisation & Chargement

Configuration de la SparkSession en mode local (local[*])
Chargement du CSV (sep=;) dans un DataFrame PySpark
541 909 lignes · 8 colonnes

2️⃣ Exploration & Prétraitement (EDA)

4 372 clients distincts avant nettoyage
Conversion UnitPrice (virgule → point) et InvoiceDate en Timestamp
Anomalies détectées et filtrées :

CustomerID NULL : 135 080 lignes
Quantity ≤ 0 : 10 624 lignes
UnitPrice ≤ 0 : 2 517 lignes


Création de TotalAmount = Quantity × UnitPrice
Dataset propre final : 4 338 clients · zéro doublon · zéro valeur nulle

3️⃣ Segmentation Client — Approche Non Supervisée

Variables RFM construites par client :

Recency — jours depuis le dernier achat
Frequency — nombre de factures distinctes
Monetary — total dépensé


Assemblage (VectorAssembler) et standardisation (StandardScaler)
Algorithme : GaussianMixture (GMM) — BisectingKMeans écarté après détection d'inertie constante
Sélection du K par indice Silhouette → K = 2 (score = 0.6084)
Filtrage des outliers extrêmes (Monetary > 25 000 €, Frequency > 100) → 4 300 clients retenus

Segments identifiés :
SegmentProfil🏆 Top AcheteursFaible récence · Haute fréquence · Fort panier moyen🔵 Clients StandardsRécence élevée · Fréquence modérée · Panier faible
4️⃣ Modélisation Supervisée — Classification

Cible : label = 1 si Monetary > 2 000 €, sinon 0
Distribution : 858 gros dépensiers (20%) · 3 442 standards (80%)
Split : 3 077 train / 1 223 test
Algorithme : RandomForestClassifier (100 arbres)
Optimisation des hyperparamètres via CrossValidator (maxDepth, numTrees)


📊 Résultats

Clustering (GaussianMixture)
MétriqueValeurAlgorithmeGaussianMixture (GMM)Nombre de clusters2Score Silhouette0.6084
Classification (RandomForest)
MétriqueValeurRappel (Recall)59.57 %Précision82.04 %AUC (après optimisation)0.918Meilleure profondeur5Meilleur nombre d'arbres100
Matrice de confusion (modèle optimisé) :
Prédit : StandardPrédit : VIPRéel : Standard963 ✅30 ❌Réel : VIP93 ❌137 ✅

🛠️ Stack Technique

CoucheOutilsTraitement distribuéApache Spark 3.5.0 · PySpark (DataFrame API)Machine Learningpyspark.ml — GaussianMixture, RandomForestClassifier, VectorAssembler, StandardScaler, CrossValidatorÉvaluationClusteringEvaluator (Silhouette), BinaryClassificationEvaluator (AUC), MulticlassClassificationEvaluatorVisualisationMatplotlib · Seaborn · Scikit-learn (ROC curve)EnvironnementGoogle Colab / Jupyter Notebook

🚀 Lancement

Sur Google Colab
python# Première cellule — installation automatique
!pip install pyspark==3.5.0 pandas seaborn scikit-learn matplotlib numpy
En local (Jupyter)
bashpip install -r requirements.txt
jupyter notebook notebook/Projet_Pyspark_VF.ipynb

📦 Dépendances

pyspark==3.5.0
pandas
matplotlib
seaborn
scikit-learn
numpy
jupyter

💡 Compétences Démontrées

Big Data : manipulation de DataFrames PySpark à grande échelle (540k+ lignes), filtres, agrégations, feature engineering
ML distribué non supervisé : GaussianMixture avec sélection du K par indice Silhouette, analyse critique des algorithmes (détection du bug BisectingKMeans)
ML distribué supervisé : RandomForest avec gestion du déséquilibre des classes et optimisation via CrossValidator
Évaluation rigoureuse : Silhouette, AUC, Précision, Rappel, matrice de confusion
Esprit critique : justification des choix algorithmiques, identification et traitement des outliers, pistes d'amélioration


📌 Pistes d'Amélioration

 Tester BisectingKMeans après révision de la pipeline de standardisation
 Ajouter une régression pour estimer la monetary_value continue (RMSE, R²)
 Gérer le déséquilibre des classes via pondération ou sur-échantillonnage
 Intégrer un pipeline Spark ML complet (Pipeline API) pour industrialisation
 Déploiement du modèle via MLflow
