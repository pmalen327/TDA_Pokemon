# Preston Malen
# March 2023

# Gudhi demo using a Pokémon data set from Kaggle
# https://www.kaggle.com/datasets/mrdew25/pokemon-database
# License: CC0: Public Domain

import gudhi as gd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

df = pd.read_csv ('Pokemon Database.csv')
df = df.dropna(axis=1)

# This list/dict stuff alone is going to get expensive lol
# Can optimize this later

# This won't be relevant until we start to interpret the data
# typeTemp = df['Primary Type'].tolist()
# typeStr = list(set(typeTemp))
# typeNum = []
# for i in range(0, len(typeStr)):
#     typeNum.append(i+1)
# types = dict(zip(typeNum,typeStr))


# id = df['Pokedex Number'].tolist()
# name = df['Pokemon Name'].tolist()
# Pokedex = dict(zip(id,name))

remove = []
for col in df.columns:
    try:
        _ = df[col].astype(int)
    except ValueError:
        remove.append(col)
        pass

# keep only numeric columns of interest
df = df[[col for col in df.columns if col not in remove]]
df = df.drop('Egg Cycle Count', axis=1)
df = df.drop('Experience Growth Total', axis=1)

data = df.to_numpy()
data = normalize(data, axis=1, norm='l1')

# beyond a max_edge_length of .05 and/or dim > 4, things are very slow
# there is a package that uses Cuda for the Rips Complex
maxEdge = .035
skeleton = gd.RipsComplex(points = data, max_edge_length = maxEdge)

maxDim = 5
st = skeleton.create_simplex_tree(max_dimension = maxDim)

# this line is more expensive than everything else combined tenfold lol
diag = st.persistence()

# should see <= max_dimension number of different colors
gd.plot_persistence_barcode(diag)
plt.show()

# TODO: persistence line graph






