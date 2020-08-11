import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]



def get_info_from_index(index):
	return df[df.index == index]["overview"].values[0]
def get_info_from_title(title):
	return df[df.title == title]["index"].values[0]




def get_trailer_from_index(index):
	return df[df.index == index]["homepage"].values[0]
def get_trailer_from_title(title):
	return df[df.title == title]["index"].values[0]


def get_date_from_index(index):
	return df[df.index == index]["release_date"].values[0]
def get_date_from_title(title):
	return df[df.title == title]["index"].values[0]

def get_vote_from_index(index):
	return df[df.index == index]["vote_average"].values[0]
def get_vote_from_title(title):
	return df[df.title == title]["index"].values[0]



def get_director_from_index(index):
	return df[df.index == index]["director"].values[0]
def get_director_from_title(title):
	return df[df.title == title]["index"].values[0]


def get_revenue_from_index(index):
	return df[df.index == index]["revenue"].values[0]
def get_revenue_from_title(title):
	return df[df.title == title]["index"].values[0]


##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
#print df.columns
##Step 2: Select Features

features = ['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:")
		print(row)

df["combined_features"] = df.apply(combine_features,axis=1)

#print "Combined Features:", df["combined_features"].head()

##Step 4: Create count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = input("Enter the movie you like:---")

## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
#movie_info  = get_info_from_title(movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))
similar_info =  list(enumerate(cosine_sim[movie_index]))
similar_trailer = list(enumerate(cosine_sim[movie_index]))
similar_date = list(enumerate(cosine_sim[movie_index]))
similar_rating = list(enumerate(cosine_sim[movie_index]))
similar_director = list(enumerate(cosine_sim[movie_index]))
similar_revenue = list(enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
sorted_movies_info = sorted(similar_info,key=lambda x:x[1],reverse=True)
sorted_movies_trailer = sorted(similar_trailer,key=lambda x:x[1],reverse=True)
sorted_movies_date =  sorted(similar_date,key=lambda x:x[1],reverse=True)
sorted_rating = sorted(similar_rating,key=lambda x:x[1],reverse=True)
sorted_director = sorted(similar_director,key=lambda x:x[1],reverse=True)
sorted_revenue = sorted(similar_revenue,key=lambda x:x[1],reverse=True)

## Step 8: Print titles of first 50 movies
i=0
#print("MODEL TRAINING ACCORDING TO DATASET COMPLETED")
print("------------------------------------------------------------------------------------Recommended Movies for you---------------------------------------------------------------------------------------------")
for element in sorted_similar_movies:
	print(i, end = '.  '),
	print(get_title_from_index(element[0]))
	print("Movie Overview- ",end = '')
	print(get_info_from_index(element[0]))
	print("homepage link-  ",end = '')
	print(get_trailer_from_index(element[0]))
	print("Release date-   ",end = '')
	print(get_date_from_index(element[0]))
	print("Rating- ",end = '')
	print(get_vote_from_index(element[0]))
	print("Director-  ",end = '')
	print(get_director_from_index(element[0]))
	print("Generated revenue-  ",end = '')
	print(get_revenue_from_index(element[0]))
	print("")
	i=i+1
	if i>50:
		break

#j=0
#print("")H
#print("------------------------------------------------------------------------------------overview of every above movie-----------------------------------------------------------------------------------------")
#for element in sorted_movies_info:
	#print("")
	#print(j, end = '.  '),
	#print(get_info_from_index(element[0]))
	#print(" ")
	#j=j+1
	#if j>50:
		#break