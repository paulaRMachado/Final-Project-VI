print("Importing")
import src.extraction as extract
import src.cleaning as clean
import src.model as model
import webbrowser

# prep data for Kmeans training and post recommendation retrieval
df_model = extract.get_dataframe("model_prep")
df_retrieve = extract.get_dataframe("df_retrieve")

print("Fitting model")
# fitting data to Kmeans model
centroid_dict, cluster_dict = model.fit_Kmeans(df_model.values, 7)
# updating centroids
centroid_dict, cluster_dict = model.train(centroid_dict, cluster_dict, df_model.values)


# get my information from Goodreads
my_data = extract.get_dataframe("my_books")
# prep the data for use in prediction
my_prep_data = clean.prep_my_data(my_data)

# set book reference
book_ref = input("Enter the index of the book you want to use as reference: ")
my_data = extract.get_dataframe("my_books")

book_title = my_data.iloc[int(book_ref)]['title']
print(f"You chose '{book_title}' as reference")

new_data_point = my_prep_data.iloc[int(book_ref)].values.tolist()

# finding book recommendation
nearest_point = model.predict_and_find_nearest(centroid_dict, cluster_dict, new_data_point)
rec_link = df_retrieve.iloc[nearest_point]["link"]

# showing recommendation
print(f" Here is a link to the book I recommend {rec_link}")
webbrowser.open(rec_link)



