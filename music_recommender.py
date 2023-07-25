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
#print(tracks.head())
#print(tracks.shape)
#print(tracks.info())

#checks for null values
#print(tracks.isnull().sum())

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

# Vectorizes
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['artist'])

# Get similar functionm
def get_similar(song_name, data):

    # Gets vector of input song
    text_array1 = song_vectorizer.transform(data[data['song_title']==song_name]['artist']).toarray()
    num_array1 = data[data['song_title']==song_name].select_dtypes(include=np.number).to_numpy()

    # Will store similarity for each row of the dataset
    similar = []
    for idx, row in data.iterrows():
        name = row['song_title']
        
        # Gets vector of current song
        text_array2 = song_vectorizer.transform(data[data['song_title']==name]['artist']).toarray()
        num_array2 = data[data['song_title']==name].select_dtypes(include=np.number).to_numpy()

        # calculates similarity
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        similar.append(text_sim + num_sim)
    
    return similar

def recommend_songs(song_name, data=tracks):
    # Base case
    if tracks[tracks['song_title'] == song_name].shape[0] == 0:
        print('This song is either not so popular or you have entered invalid_name.\n Some songs you may like:\n')

        for song in data.sample(n=5).values:
            print(song)
        return
    
    data['similarity_factor'] = get_similar(song_name, data)

    data.sort_values(by=['similarity_factor', 'danceability'],
                     ascending= [False, False],
                     inplace= True)
    
    print(data[['song_title', 'artist']][15:16])

recommend_songs('Redbone')