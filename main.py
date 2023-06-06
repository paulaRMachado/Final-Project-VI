import src.extraction as extract
import src.cleaning as clean
import src.visualizing as viz
import src.model as model

# retrieve original database from Goodreads
df = extract.get_dataframe("goodreads_books")
# clean database for visualizations
df_clean = clean.basic_clean(df)
# save cleaned data
clean.save_dataframe(df_clean,"books_clean")
# prep data for Kmeans training and post recommendation retrieval
df_model, df_retrieve = clean.prep_model(df)

"""
No plotting needed

# visualizing final correlation map
viz.matrix_corr(df_model,"correlation_heatmap_final")
# visualizing elbow chart
viz.elbow_method(df_model)
# visualizing silhouette plot
viz.silhouette_plot(df_model)
# visualizing davies boulding plot
viz.davies_boulding_plot(df_model)
# visualizing blobs
viz.kmeans_plot_blobs()
"""

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

new_data_point = my_prep_data.iloc[int(book_ref)].values.tolist()
# finding book recommendation
nearest_point = model.predict_and_find_nearest(centroid_dict, cluster_dict, new_data_point)

df_retrieve.iloc[nearest_point]
my_data.iloc[int(book_ref)]



