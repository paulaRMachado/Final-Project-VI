import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler
from src.visualizing import matrix_corr 

sia = SentimentIntensityAnalyzer()

def count_years(text):
    """
    This function counts the appearences of an expression with 4 numbers 
    and replaces the award names with the number of awards.
    
    :args:
        text: A string of any size
    :returns:
        the number of times the regex pattern was found
    """
    if isinstance(text, str):
        years = re.findall(r"\d{4}", text)
        return len(years)
    else:
        return 0
    
def extract_genre(text):
    """
    This function takes the text form the column and removes the number of votes on genre and the word user
    """
    if isinstance(text, str):
        genre = re.sub(r'\d+|\buser[s?]\b', '', text)
        genre = genre.strip()
        return genre
    else:
        return ''


def coleman_liau_index(text):
    """
    This function calculates the Coleman-Liau Index wich provides an estimate of the grade level 
    required to understand the text, based on characters instead of syllables.

    Low Readability: Coleman-Liau Index score below 6
    Typically suitable for elementary school-level readers.

    Medium Readability: Coleman-Liau Index score between 6 and 10
    Generally appropriate for an average adult audience or readers at a high school level.

    High Readability: Coleman-Liau Index score above 10
    Typically requires a higher level of reading comprehension 
    and may be more suitable for advanced readers or academic texts
    """
    num_letters = sum(c.isalpha() for c in text)
    words = word_tokenize(text)
    num_words = len(words)
    num_sentences = len(sent_tokenize(text))

    if num_words == 0:
        return 0

    average_letters_per_word = num_letters / num_words * 100
    average_sentences_per_word = num_sentences / num_words * 100

    grade_level = 0.0588 * average_letters_per_word - 0.296 * average_sentences_per_word - 15.8

    return round(grade_level, 2)

def basic_clean(df):
    """
    this function prepares the dataframe for visualisations, droping some columns separating genre cetegories
    It also calculates sentiment scores from the description and readability index.
    """
    # droping unnecessary columns
    df.drop(columns=["characters","settings","id","publisher","original_title","worldcat_redirect_link", "cover_link", "author_link","amazon_redirect_link", "worldcat_redirect_link","isbn", "isbn13", "asin"], inplace = True)
    df.drop_duplicates(inplace=True)

    # checking title formating
    df['title'] = df['title'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['title'] = [i.strip() for i in df['title']]
    df = df[df['title'].notnull() & (df['title'] != '')]
    titles_to_remove = [":", "-", "1", "2","3"]
    df = df[~df['title'].isin(titles_to_remove)]

    # checking author formating
    df['author'] = df['author'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['author'] = [i.strip() for i in df['author']]

    # series becomes a boolean
    df["series"] = df["series"].apply(lambda x: False if pd.isnull(x) else True)
    
    # defining year of publishing
    df["year_published"] = [re.findall(r"\d{4}", i)[-1] if (isinstance(i, str) and re.findall(r"\d{4}", i)) else "0" for i in df["date_published"]]
    df["year_published"] = [0 if int(i) < 1000 else i for i in df["year_published"]]
    df["year_published"] = df["year_published"].astype(int)
    
    # counting awards
    df["awards"] = df["awards"].fillna(0).apply(count_years)

    # cleaning genre column
    df["genres"] = df["genre_and_votes"].apply(extract_genre)
    df[['subgenre', 'subgenre1', 'subgenre2']] = df['genres'].str.split(',', expand=True, n=2)
    df['subgenre'] = [i.replace("user", "").strip() for i in df['subgenre']]

    condition1 = df['subgenre'] == "Science Fiction"
    condition2 = df.subgenre.str.contains("Fiction")
    df.loc[~condition1 & condition2, 'subgenre'] = "Fiction"

    # Define the mappings for genre
    genres = {"Children": "Childrens","Kids":"Childrens", "Horror":"Horror", "Adventure":"Fiction","Sequential":"Sequential Art","Comics":"Sequential Art","Economics":"Business","Philosophy":"Philosophy", "Sports":"Sports",
          "Fairy":"Fairy Tales", "Fantasy":"Fantasy", "Science Fiction":"Science Fiction", "Thriller":"Thriller/Mystery","Young Adult":"Young Adult","Anthologies":"Anthologies","Writing":"Writing",
          "Romance":"Romance","History":"History","World War":"History","Mystery":"Thriller/Mystery", "Religion":"Religion","Paranormal":"Horror","Cultur":"Cultural","Africa":"Cultural",
          "Non":"Nonfiction","Literature":"Literature", "Health":"Health", "Media Tie":"Fiction", "Erotica": "Adult","Design":"Art","Music":"Art" ,"Food":"Food and Drinks","Academic":"Academic",
          "Animals":"Animals","Amish":"Cultural","Werewolves":"Fantasy", "Business":"Business", "Science-":"Science","Occult":"Religion","Magical":"Fantasy","Fem":"Sociology",
          "Game":"Games","Psychology":"Health","Witchcraft":"Fantasy","Christian":"Religion","Spirituality":"Religion","Alcohol":"Food and Drinks","LGBT":"Sociology","Plays":"Plays",
          "Holiday":"Cultural","Self Help":"Health","Manga":"Sequential Art","Crafts":"Art", "Anthropology":"Sociology","Gender":"Sociology","GLBT":"Sociology","Biology":"Science",
          "Historical":"History","Suspense":"Thriller/Mystery","Folk":"Cultural","Fat":"Health","Sociology":"Sociology","Apocalyptic": "Fiction","New Adul":"Young Adult","Crime":"Thriller/Mystery",
          "Photography":"Art","Dark":"Horror","Classics":"Classics","Travel":"Travel","Biography":"Biography", 'Humor':'Humor',"Diary":"Biography","Poetry":"Poetry","Novel":"Novels",
          "Law":"Law","Environment":"Environment", "Teaching":"Education","Textbooks":"Education", "Contemporary":"Contemporary","Adult -":"Adult","Pornography":"Adult", "Humanities":"Humanities",
          "Linguistics":"Humanities"
        }
    search_dict = {k.lower(): v for k, v in genres.items()}
    df["subgenre"] = df["subgenre"].map(lambda x: search_dict.get(next((i for i in search_dict if i in str(x).lower()), x), x))
  

    # Define the mappings for greater classifications
    class_mapping = {'Fiction': 'Literature','Fantasy': 'Literature','Romance': 'Literature','Young Adult': 'Literature','Horror':'Literature','Sequential Art':'Literature',
                    'Thriller/Mystery':'Literature','Classics':'Literature','Science Fiction':'Literature','Drama':'Literature','Anthologies-Collections':'Literature',
                    'Novels':'Literature','Drama':'Literature','Poetry':'Literature','Gothic':'Literature','Book Club':'Literature','Adult':'Literature','Novella':'Literature',
                    'Futuristic':'Literature','Humor':'Literature','Humor':'Literature','Novel':'Literature','Romantic':'Literature','Epic':'Literature','Modern':'Literature',
                    'Westerns':'Literature','Love':'Literature','Tragedy':'Literature','Anthologies':'Literature','Nobel Prize':'Literature','Action':'Literature','Contemporary -':'Literature','Space':'Literature',
                    'Nonfiction':'Nonfiction','History':'Nonfiction','Religion':'Nonfiction','Self Help':'Nonfiction','Biography':'Nonfiction','Cultural':'Nonfiction',
                    'Food and Drinks':'Nonfiction', 'Sports':'Nonfiction','Philosophy':'Nonfiction','Contemporary':'Nonfiction','Politics':'Nonfiction',
                    'True Story':'Nonfiction','Leadership':'Nonfiction','Travel':'Nonfiction','Productivity':'Nonfiction','War':'Nonfiction',
                    'Currency-Money':'Nonfiction','Diary-Journal':'Nonfiction','Parenting':'Nonfiction','Esoterica-Alchemy':'Nonfiction','Sexuality':'Nonfiction','Retellings':'Nonfiction',
                    'Criticism':'Nonfiction','American-Southern':'Nonfiction','United States':'Nonfiction','Parenting-Adoption':'Nonfiction','Autobiography-Memoir':'Nonfiction',
                    'Inspirational':'Nonfiction','Biography-Autobiography':'Nonfiction','Marriage':'Nonfiction','Combat-Martial Arts':'Nonfiction','Nature-Outdoors':'Nonfiction',
                    'Football':'Nonfiction','Soccer':'Nonfiction','New York':'Nonfiction','Relationships':'Nonfiction','Warfare-Fighters':'Nonfiction','Prayer':'Nonfiction',
                    'Business': 'Academic','Economics': 'Academic','Law':'Academic','Medical':'Academic','Animals':'Academic','Sociology':'Academic','Science':'Academic',
                    'Health':'Academic','Academic':'Academic','Buisness':'Academic','Education':'Academic','Language-Communication':'Academic',
                    'Artificial Intelligence':'Academic','Race-Anti Racist':'Academic','Illness-Disease':'Academic',
                    'Writing':'Academic','Social Issues-Class':'Academic','Mythology-Greek Mythology':'Academic',
                    'Humanities':'Academic','Reference-Research':'Academic','Linguistics-Semiotics':'Academic','Reference':'Academic','Politics-Political Science':'Academic','Disability':'Academic',
                    'Computers-Internet':'Academic','Environment':'Academic','Polyamorous-Reverse Harem':'Academic','Sexuality-Sex Work':'Academic',
                    'Art':'Art','Music':'Art','Art-Photography':'Art','Architecture':'Art','Couture-Fashion':'Art','Games':'Art',
                    'Childrens':'Childrens',                   
                    '':'Unkown'
    }
    
    df['classification'] = df['subgenre'].map(class_mapping).fillna('Other')

    genre_counts = df.subgenre.value_counts()
    df.subgenre = df.subgenre.replace(genre_counts[genre_counts <= 20].index, "Other")
        
    # Calculate the compound sentiment score for each description
    df['compound_score'] = df['description'].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Calculate complexity
    df['coleman_liau_index'] = df['description'].fillna('').apply(coleman_liau_index)

    #final drop
    df.drop(columns=["date_published","genre_and_votes","genres","description","subgenre1","subgenre2"], inplace=True)
    
    return df

def prep_model(df):
    """
    This function prepares the data for going through the model.

    :args:
    df: a dataframe to be preped for clustering

    :returns:
    df_model: dataframe ready for training
    """
    # droping unnecessary columns
    df.drop(columns=["review_count","title","author","recommended_books","books_in_series"], inplace = True)

    ## transforming columns so model can understand
    scaler = StandardScaler()
    df["rating_count_stand"] = scaler.fit_transform(df["rating_count"].values.reshape(-1, 1))
    
    # correcting genre type
    dict_genre = {"Fiction":0, "Fantasy":1, "Romance":2, "Young Adult":3, "Thriller/Mystery":4,
              "Sequential Art":5,"Science Fiction":6,"Classics":7,"Horror":8,"Poetry":9,"Novels":10,
              "Humor":11,"Adult":12,"Westerns":13,"Other":14,"Drama":15,"Anthologies":16
    }
    df["subgenre"] = df["subgenre"].replace(dict_genre)

    # drop
    df_retrieve = df[df["classification"] == "Literature"].copy()
    df_retrieve.drop(columns=["rating_count","classification"], inplace=True)
    
    matrix_corr(df_retrieve.drop("link", axis = 1),"correlation_heatmap_prev")
    df_retrieve.drop(columns=["five_star_ratings","four_star_ratings","three_star_ratings","two_star_ratings","one_star_ratings"], inplace=True)
    df_retrieve.dropna(inplace=True)

    df_model = df_retrieve.drop("link", axis=1).copy()
  
    return df_model, df_retrieve

def prep_my_data(df):
    """
    This function prepares the data for going through the model.
    :args:
    df: a dataframe to be preped for clustering
    """

    # correcting genre type
    dict_genre = {"Fiction":0, "Fantasy":1, "Romance":2, "Young Adult":3, "Thriller/Mystery":4,
              "Sequential Art":5,"Science Fiction":6,"Classics":7,"Horror":8,"Poetry":9,"Novels":10,
              "Humor":11,"Adult":12,"Westerns":13,"Other":14,"Drama":15,"Anthologies":16
    }
    df["subgenre"] = df["subgenre"].replace(dict_genre)

    # Calculate the compound sentiment score for each description
    df['compound_score'] = df['description'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Calculate complexity
    df['coleman_liau_index'] = df['description'].apply(coleman_liau_index)

    # transforming columns so model can understand
    scaler = StandardScaler()
    df["rating_count_stand"] = scaler.fit_transform(df["rating_count"].values.reshape(-1, 1))

    df["average_rating"] = df["average_rating"].str.replace(",", ".")
    df["average_rating"] = df["average_rating"].astype(float)

    df.drop(columns=["description","rating_count","title"], inplace=True)
  
    return df


def save_dataframe(df, name):
    """
    This function save a dataframe under a given name in the data folder.
    :args:
    df: dataframe to be saved
    name: name to save under
    """
    df.to_csv(f'data/{name}.csv', index=False)
    pass