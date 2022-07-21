import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 20)


df = pd.read_csv('Books.csv', encoding = 'unicode_escape', low_memory=False)

def Random_Recommendations():
    random_df= df.sample(n=100)
    return  random_df;

def Best_Dataframe():
    averageRating = df['Rating'].mean()
    minimum_votes =df['RatingDistTotal'].quantile(0.90)
    q_Books = df.copy().loc[df['RatingDistTotal'] >= minimum_votes]
    
    def weighted_rating(x, m= minimum_votes, c= averageRating):
        Votes_number = x['RatingDistTotal']
        movie_rating = x['Rating']
        # Calculation based on the IMDB formula
        return (Votes_number/(Votes_number+minimum_votes) * movie_rating) + (minimum_votes/(Votes_number+minimum_votes) * averageRating)
    
    q_Books['score'] = q_Books.apply(weighted_rating, axis=1)
    q_Books = q_Books.sort_values('score', ascending=False)
    q_Books = q_Books.reset_index(drop=True)
    return q_Books;

def Best_Recommendations():
    q_Books = Best_Dataframe()
    best_df = q_Books.head(100)
    return best_df;

q_Books = Best_Dataframe()
tfidf = TfidfVectorizer(stop_words='english')
q_Books['Description'] = q_Books['Description'].fillna('')
tfidf_matrix = tfidf.fit_transform(q_Books['Description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(q_Books.index, index=q_Books['Name']).drop_duplicates()

def Content_Recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommended_books = q_Books.iloc[movie_indices]
    return recommended_books;

def Search(text):
    result = df[df.apply(lambda row: row.astype(str).str.contains(text, case=False).any(), axis=1)]
    return result;


"""Test Cases"""
print("1- Random Recommendations")
# 1- Random Recommendations
Random = Random_Recommendations()
print (Random)

print("2- Best Books Recommendations")
# 2- Best Books Recommendations
Best = Best_Recommendations()
print (Best)

print("3- Similar Content Recommendations")
# 3- Similar Content Recommendations (You can change the book name if you want)
Similar = Content_Recommendations("Harry Potter and the Order of the Phoenix")
print (Similar)

print("4- Search")
# 4- Search for any book using any search term (You can change the search term if you want)
Search_List = Search("Batman")
print (Search_List)