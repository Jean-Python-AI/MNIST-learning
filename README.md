# MNIST — classificateur de chiffres manuscrits

Ce projet entraîne « à la main » un MLP, avec NumPy et sans framework de réseau de neurones, à reconnaître les chiffres manuscrits de **0 à 9**.

## Vue d'ensemble

Le chemin suivi par une image est :

```text
PNG 28 × 28 → vecteur de 784 pixels → MLP → 10 scores → chiffre prédit
```

L'architecture actuellement définie est :

```text
[784, 100, 140, 80, 10]
```

- `784` : les $28\times28$ pixels de l'image ;
- `100`, `140`, `80` : les couches cachées ;
- `10` : une sortie par chiffre possible.

Les fichiers importants du projet sont `main.py`, `AI/MLP.py` et `Dataset/Dataset.py`.

## 1. Préparer les données

Les images sont organisées ainsi :

```text
Dataset/
├── training/0 … 9/
└── testing/0 … 9/
```

Le nom du **dossier** est le label : une image placée dans `training/7/` représente le chiffre 7. Le nom du fichier PNG est seulement un identifiant ; il ne sert pas à déterminer le chiffre.

`DATA` lit chaque image avec Pillow, la convertit en niveaux de gris, la redimensionne en $28\times28$, puis normalise chaque pixel :

$$
x_{\text{normalisé}}=\frac{x_{\text{pixel}}}{255}.
$$

Chaque pixel appartient alors à $[0,1]$. L'image est ensuite aplatie :

$$
(28,28)\longrightarrow(784,).
$$

Pour une image du chiffre 3, le label *one-hot* est :

$$
Y=[0,0,0,1,0,0,0,0,0,0].
$$

Les tableaux du dataset ont donc les formes :

```text
X : (nombre_d_images, 784)
Y : (nombre_d_images, 10)
```

## 2. Propagation avant

Chaque couche $c$ possède :

- $W^{[c]}$, la matrice de poids ;
- $b^{[c]}$, le vecteur de biais ;
- $Z^{[c]}$, le résultat du calcul linéaire ;
- $A^{[c]}$, l'activation transmise à la couche suivante.

Pour un exemple, les vecteurs sont des colonnes. Si la couche précédente possède $p$ neurones et la couche courante $q$ neurones :

```text
A[c-1] : (p, 1)
W[c]   : (q, p)
b[c]   : (q, 1)
Z[c]   : (q, 1)
```

Le calcul linéaire est :

$$
Z^{[c]}=W^{[c]}A^{[c-1]}+b^{[c]}.
$$

Le projet applique actuellement la sigmoïde :

$$
\sigma(Z)=\frac{1}{1+e^{-Z}},
\qquad
A^{[c]}=\sigma\!\left(Z^{[c]}\right).
$$

La sigmoïde transforme chaque valeur en un nombre compris entre 0 et 1.

Dans le code, un batch est traité exemple par exemple, puis les activations sont regroupées. Conceptuellement, un batch de taille $m$ produit :

```text
A[c] : (m, nombre_de_neurones_de_la_couche, 1)
```

## 3. Mesurer l'erreur : la log-loss

La version actuelle utilise une sigmoïde sur chaque sortie et une **binary cross-entropy** appliquée terme à terme :

$$
L=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10}
\left[
Y_{i,k}\log(A_{i,k})+(1-Y_{i,k})\log(1-A_{i,k})
\right].
$$

`np.clip(A, 10^{-15}, 1-10^{-15})` évite les calculs impossibles, notamment $\log(0)$.

La loss est faible lorsque la sortie du bon chiffre est proche de 1 et que les autres sont proches de 0.

> [!warning] Classification à 10 classes
> Les chiffres sont mutuellement exclusifs : une image ne peut représenter qu'un seul chiffre. Le choix standard est donc **softmax sur la dernière couche + categorical cross-entropy**. Le projet actuel utilise 10 sigmoïdes + binary cross-entropy ; `argmax` permet tout de même de choisir une classe, mais ce n'est pas la formulation la plus naturelle.
>
> Ne pas appliquer un softmax *après* une sigmoïde uniquement pour l'affichage : la sigmoïde a déjà comprimé les scores dans $[0,1]$. Avec 10 sorties, le softmax de ces valeurs peut difficilement être très confiant. Si le projet passe au softmax, il doit recevoir directement les derniers $Z^{[n]}$, les *logits*.

## 4. Backpropagation : apprendre les paramètres

Pendant la propagation avant, le réseau mémorise $A^{[c]}$ et $Z^{[c]}$. Il remonte ensuite l'erreur de la dernière couche vers la première : c'est la backpropagation.

Pour une dernière couche sigmoïde avec binary cross-entropy, le gradient se simplifie en :

$$
dZ^{[n]}=\frac{A^{[n]}-Y}{m}.
$$

Pour une couche cachée :

$$
dA^{[c]}=\left(W^{[c+1]}\right)^T dZ^{[c+1]},
$$

$$
dZ^{[c]}=dA^{[c]}\odot A^{[c]}\left(1-A^{[c]}\right),
$$

où $\odot$ est une multiplication terme à terme. Le $W^T$ est un **produit matriciel** : il ramène l'erreur de la couche suivante vers les neurones de la couche actuelle.

Pour chaque exemple $i$, le gradient des poids est un produit extérieur :

$$
dW_i^{[c]}=dZ_i^{[c]}\left(A_i^{[c-1]}\right)^T.
$$

Les gradients du batch sont additionnés :

$$
dW^{[c]}=\sum_i dW_i^{[c]},
\qquad
db^{[c]}=\sum_i dZ_i^{[c]}.
$$

Dans ce projet, le facteur $1/m$ est déjà inclus dans $dZ^{[n]}$. Il ne faut donc pas redédiviser les gradients par $m$ : une moyenne doit apparaître **une seule fois**.

La descente de gradient met ensuite à jour les paramètres :

$$
W^{[c]}\leftarrow W^{[c]}-\alpha dW^{[c]},
$$

$$
b^{[c]}\leftarrow b^{[c]}-\alpha db^{[c]},
$$

avec $\alpha=0{,}01$ dans le code.

## 5. Entraîner sans saturer la mémoire

Une époque signifie que le modèle a vu toutes les images d'entraînement une fois.

1. `rng.permutation(len(X))` mélange tous les indices sans doublon.
2. Les indices sont découpés en mini-batches de 32 images.
3. Chaque batch suit le cycle : propagation avant $\rightarrow$ loss $\rightarrow$ backpropagation.
4. Le modèle est sauvegardé à la fin de l'époque.

Le mélange évite que le réseau voie d'abord toutes les images de 0, puis toutes les images de 1, etc. Un exemple n'est pas répété pendant une époque ; il revient naturellement lors de l'époque suivante, dans un autre ordre.

## 6. Prédire et afficher

En mode *SHOW*, 12 images de test différentes sont choisies :

```python
indices = rng.choice(len(X), size=12, replace=False)
```

La classe prédite est celle dont le score est maximal :

$$
\hat y=\operatorname*{argmax}_{k\in\{0,\ldots,9\}} A_k^{[n]}.
$$

Le vecteur de 784 valeurs est remis en image pour Matplotlib :

```python
image = X_show[i].reshape(28, 28)
```

Le titre compare `Prédit` et `Vrai`. Vert signifie que la prédiction est correcte ; rouge indique une erreur. En mode *SHOW*, il ne faut appeler ni `LogLoss()` ni `BackProp()` : l'affichage ne doit ni entraîner le modèle ni modifier ses paramètres.

## 7. Sauvegarder et reprendre l'entraînement

`np.savez(..., **self.PARAMETRES)` enregistre `W1`, `b1`, `W2`, `b2`, etc. dans `AI/models/<nom>.npz`.

```text
Niew=True  → crée de nouveaux poids aléatoires et écrase le fichier au départ.
Niew=False → charge les poids existants.
```

Pour afficher un modèle déjà entraîné, il faut donc utiliser `Niew=False`. Sauvegarder une fois par époque est préférable à une sauvegarde après chaque mini-batch.

## Points de contrôle

- Les données et les labels ont les formes attendues.
- Chaque batch contient des exemples mélangés et distincts.
- La loss diminue sur l'entraînement sans que le mode *SHOW* ne change les poids.
- Le modèle chargé avec `Niew=False` donne les mêmes prédictions avant tout nouvel entraînement.
