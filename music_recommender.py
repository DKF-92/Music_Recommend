# Import dependencies/libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb 

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# Reads dataset
tracks = pd.read_csv('Music_Recommend\musicdataset.csv')
print(tracks.head())
print(tracks.shape)
print(tracks.info())

#checks for null values
print(tracks.isnull().sum())

#drops unneeded columns
tracks = tracks.drop(['id'], axis=1)

"""""
# use t-SNE for visualization of high dimensional data
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit()
plt.figure(figsize= (7, 7))
plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.show()
"""""

# Looks for non unique track names (we will notice some duplicates)
print(tracks['song_title'].nunique(), tracks.shape)

# drops the duplicate song names
tracks.drop_duplicates(subset=['song_title'], keep='first', inplace=True)
print(tracks.shape)