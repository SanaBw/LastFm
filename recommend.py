import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tkinter import *
import math
from fuzzywuzzy import fuzz


play_count = pd.read_table('plays.tsv',
                        header = None, nrows = 2e7,
                        names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                        usecols = ['users', 'artist-name', 'plays'])
users = pd.read_table('usersha1-profile.tsv',
                        header = None,
                        names = ['users', 'gender', 'age', 'country', 'signup'],
                        usecols = ['users', 'country'])


if play_count['artist-name'].isnull().sum() > 0:
    play_count = play_count.dropna(axis = 0, subset = ['artist-name'])

total_plays = (play_count
               .groupby(by = ['artist-name'])['plays']
               .sum()
               .reset_index()
               .rename(columns = {'plays': 'total_artist_plays'})
               [['artist-name', 'total_artist_plays']]
              )


total_count = play_count.merge(total_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')



threshold = 7929
popular_plays = total_count.query('total_artist_plays >= @threshold')



popular_plays_final = popular_plays.merge(users, left_on = 'users', right_on = 'users', how = 'left')
popular_plays_final = popular_plays_final.query('country == \'United States\'')





popular_plays_final = popular_plays_final.drop_duplicates(['users', 'artist-name'])  


final = popular_plays_final.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
final_sparse = csr_matrix(final.values)



from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(final_sparse)



def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
   
    query_index = None
    ratio_tuples = []
    
    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
    except:
        label3=Label(root, text="Your artist didn't match any artists in the data. Try again")
        label3.pack()
        return None
    
    distances, indices = knn_model.kneighbors(artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors = k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            label4=Label(root, text='Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index]))
            label4.pack()
        else:
            label4=Label(root, text='{0}, with distance of {1}'.format(artist_plays_matrix.index[indices.flatten()[i]], math.ceil((distances.flatten()[i])*100)/100))
            label4.pack()

    return None


def callRecommend(event):
    artist = name.get()
    print_artist_recommendations(artist, final, model_knn, k=10)


root = Tk()
root.geometry('300x600')
root.grid()
root.title("Artist Recommender")
label1 = Label(root, text="Enter an artist")
label1.pack()
name = Entry(root)
name.pack()

button_1 = Button(root, text="Recommend")
button_1.bind("<Button-1>", callRecommend)
button_1.pack()


root.mainloop()
