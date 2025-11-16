# Rapport de Projet

## Le Jeu de Données

### Données des Artistes (artists.csv)

-   Noms des artistes
-   Total de streams (en tant qu'artiste principal)
-   Nombre de morceaux publiés
-   Morceaux avec plus d'1 milliard de streams
-   Morceaux avec plus de 100 millions de streams
-   Nombre de collaborations en featuring

### Données des Morceaux (tracks.csv)

-   Informations de base : nom du morceau, artiste(s), genre, score de popularité
-   Caractéristiques audio que Spotify calcule pour chaque chanson :

| Caractéristique  | Ce qu'elle mesure                              |
| ---------------- | ---------------------------------------------- |
| Danceability     | Facilité à danser (échelle 0-1)                |
| Energy           | Intensité et activité de la chanson (0-1)      |
| Loudness         | Volume global en décibels                      |
| Speechiness      | Proportion de paroles vs chant (0-1)           |
| Acousticness     | Instruments acoustiques vs électroniques (0-1) |
| Instrumentalness | Pas de voix vs voix dominante (0-1)            |
| Valence          | Positivité/joie de la chanson (0-1)            |
| Tempo            | Vitesse en battements par minute (BPM)         |

## Nettoyage des Données

### Problèmes rencontrés :

1. **Nombres désordonnés** : Les comptages de streams avaient des virgules dedans. Par exemple, Drake avait "50,162,292,808" streams, ce que Python n'aime pas, donc nous avons dû les enlever et les convertir en nombre (50162292808) correctement.

2. **Valeurs manquantes** : Certaines chansons n'avaient pas de données pour les caractéristiques audio. Nous les avons remplies en utilisant la valeur médiane pour chaque caractéristique (la médiane fonctionne mieux que la moyenne car elle n'est pas affectée par les valeurs extrêmes).

3. **Noms de colonnes** : Il y avait des noms de colonnes problématiques. Par exemple, "Artist Name" au lieu d'un nom simple, ce qui compliquait l'accès aux données. Nous avons nettoyé et renommé ces colonnes.

Nous avons aussi créé de nouvelles caractéristiques :

-   Scores de popularité normalisés (échelle 0-100) basés sur les comptages de streams
-   Nombre d'artistes par morceau (solo vs collaborations)
-   Durée des chansons convertie de millisecondes en minutes pour la lisibilité

## Méthodes d'Analyse

### 1. Clustering K-Means

K-Means était la première méthode de clustering que nous avons essayée. L'idée est de regrouper les chansons qui ont des caractéristiques audio similaires.

**Comment nous l'avons fait :**

D'abord, nous avons dû standardiser les données. C'est important car les caractéristiques ont des échelles différentes - le tempo va de 60 à 200 environ, mais la danceability n'est que de 0 à 1. Si vous ne standardisez pas, les nombres plus grands dominent l'analyse.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(tracks_df[audio_features])
```

Ensuite, nous avons utilisé la "méthode du coude" pour déterminer combien de clusters utiliser. Nous avons essayé différentes valeurs de K (de 1 à 10) et tracé les résultats. Le graphique ressemble à un coude, et là où il se plie est généralement la meilleure valeur de K. Pour ce jeu de données, K=4 semblait bien fonctionner.

```python
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

**Ce que nous avons trouvé :**

Les 4 clusters qui ont émergé avaient du sens musicalement :

-   **Cluster 0 - Chansons Chill/Acoustiques** : Musique folk, chansons à la guitare acoustique.
-   **Cluster 1 - Énergique/Pop** : Musique pop radio typique.
-   **Cluster 2 - Vocal/Mainstream** : La plupart des morceaux populaires tombaient ici.
-   **Cluster 3 - Électronique/Instrumental** : EDM et musique électronique.

Fait intéressant, le cluster pop énergique avait la popularité moyenne la plus élevée.

**Défi de visualisation :**

On ne peut pas vraiment tracer des données à 8 dimensions, donc nous avons utilisé PCA (Analyse en Composantes Principales) pour les réduire à 2 dimensions pour la visualisation. PCA trouve les deux directions qui capturent le plus de variation dans les données. Ce n'est pas parfait mais ça permet de voir les clusters sur un graphique 2D.

### 2. Clustering DBSCAN

DBSCAN est assez différent de K-Means. Au lieu de forcer tout dans K groupes, il cherche des régions denses de données et appelle tout le reste "bruit" ou valeurs aberrantes.

Nous avons défini deux paramètres :

-   `eps=0.5` : Quelle proximité les points doivent avoir pour compter comme voisins
-   `min_samples=10` : Combien de voisins vous devez avoir pour être considéré comme un point "core"

**Pourquoi utiliser DBSCAN ?**

K-Means suppose que les clusters sont circulaires/sphériques, mais DBSCAN peut trouver des clusters de formes bizarres. De plus, il détecte automatiquement les valeurs aberrantes, ce qui est utile pour trouver des chansons vraiment inhabituelles.

**Résultats :**

DBSCAN a généralement trouvé 2-5 clusters principaux et a marqué environ 20-40% des chansons comme bruit. Au début, nous pensions que c'était beaucoup, mais ça fait sens en fait - il y a plein de chansons expérimentales ou de niche qui ne correspondent pas aux patterns normaux.

La détection de bruit est en fait utile car elle met en évidence les chansons qui sont vraiment différentes de tout le reste.

### 3. Analyse de Corrélation

Nous voulions voir quelles caractéristiques audio sont liées les unes aux autres. Les mesures de corrélation vont de -1 (relation négative parfaite) à +1 (relation positive parfaite).

**Corrélations fortes que nous avons trouvées :**

-   **Énergie et Loudness** (r ≈ 0.76) : Fait totalement sens - les chansons fortes semblent plus énergiques.
-   **Énergie et Acousticité** (r ≈ -0.73) : Les chansons acoustiques tendent à être moins énergiques. La production électronique augmente l'énergie.
-   **Valence et Danceability** (r ≈ 0.48) : Les chansons joyeuses sont plus dansantes.
-   **Speechiness et Instrumentalité** (négatif) : Évidemment - plus de paroles signifie moins de contenu instrumental.

**Découverte intéressante sur la popularité :**

La popularité n'était pas fortement corrélée avec une seule caractéristique audio (toutes les corrélations en dessous de 0.3). Cela suggère que ce qui rend une chanson populaire n'est pas juste une question de caractéristiques audio - c'est probablement plus une question de marketing, de célébrité de l'artiste, de timing, de médias sociaux, etc.

### 4. Analyse de Genre

Nous avons regardé quels genres sont les plus communs et lesquels sont les plus populaires.

**Genres les plus communs dans le jeu de données :**

1. Pop
2. Rock
3. Hip-Hop/Rap
4. Électronique/EDM
5. R&B

**Genres les plus populaires** (par score de popularité moyen) :

1. Pop Film
2. Pop
3. Progressive House

Nous avons aussi calculé les caractéristiques audio moyennes pour chaque genre, ce qui montre que les genres ont des "signatures" distinctes :

-   Pop : Haute danceability, haute valence (joyeux), énergie modérée
-   Rock : Haute énergie, danceability plus faible
-   Électronique : Très faible acousticité, haute instrumentalité
-   Folk/Acoustique : Haute acousticité, énergie plus faible
