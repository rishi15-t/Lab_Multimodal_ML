import os
import json
import pandas as pd
from PIL import Image
import nltk
import matplotlib.pyplot as plt 
import seaborn as sns
#%matplotlib inline
import numpy as np
import random




'''
USAGE:

data = Read(folder_path)
ViewDataDistribution(data)
data.to_pickle("./mm_imdb.pkl")

'''
def Read(path) :
    im_files = os.listdir(path)
    current_directory = os.getcwd()
    os.chdir(path)
   
    #images_cv will return list of numpy arrays containing RGB values
    images_cv = []
    for a in im_files: # Iterate over a copy of the list
        if a.endswith(".jpeg"):
            images_cv.append(Image.open(a).resize((256,256)).convert('RGB'))

    images_cv = pd.DataFrame(images_cv, columns = ['image'])
    
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    jsons_data = pd.DataFrame(columns=['genres', 'plot', 'fileID'])
    for index, js in enumerate(json_files):
        with open(os.path.join(path, js)) as json_file:
            json_text = json.load(json_file)
            genres = json_text['genres']
            plot = json_text['plot'][0]
            fileID = js.rstrip(".json")
            #jsons_data will return a dataframe with genres,file names and plot
            jsons_data.loc[index] = [genres,plot,fileID]
           
    os.chdir(current_directory)
   
    return (pd.concat([jsons_data, images_cv], axis=1))




def ViewDataDistribution(data):
    
    all_genres = sum(data["genres"],[])
    unique_genres = (set(all_genres))
    all_genres = nltk.FreqDist(all_genres)
    all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})
    plt.figure(figsize=(15,12)) 
    ax = sns.barplot(data=all_genres_df, x= "Count", y = "Genre") 
    for p in ax.patches:
        ax.annotate("%d" % p.get_width(), (p.get_x() + p.get_width(), p.get_y() + 0.5), xytext=(5,0), textcoords='offset points')
    
    plt.show()





def RemoveGenresFromData(data):
    
    genres_to_remove = ["Adult","News","Talk-Show","Reality-TV"]
    data_genres_removed = data[~np.array([bool(set(genre) & set(genres_to_remove)) for genre in data["genres"]])] 
    data_genres_removed = data_genres_removed.reset_index(drop=True)
    
    return(data_genres_removed)




def MoviesPerGenre(data):
    
    data_single_genre = pd.DataFrame({
                                  col:np.repeat(data[col].values, data["genres"].str.len())
                                  for col in data.columns.drop("genres")}
                                ).assign(**{"genres":np.concatenate(data["genres"].values)})[data.columns]
    fileID_by_genre = data_single_genre.groupby("genres")["fileID"].apply(list).reset_index(name='fileIDs')
    
    return(fileID_by_genre)




'''
USAGE:

samples = SamplingByCount(data,338)
ViewDataDistribution(samples)
samples.to_pickle("./mm_imdb_sampled.pkl")

'''
def SamplingByCount (data, count = 330) :
    
    data_subset = RemoveGenresFromData(data)
    fileID_by_genre = MoviesPerGenre(data_subset)
    samples_fileIDs = []

    for index, row in fileID_by_genre.iterrows():
    
        samples_fileIDs.extend(random.sample(row["fileIDs"],count))
        
    data_sampled = data_subset[data_subset["fileID"].isin(samples_fileIDs)]
    
    return (data_sampled)