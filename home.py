import streamlit as st
import streamlit.components.v1 as components
import src.extraction as extract
import src.model as model
import webbrowser

st.title("Book recommendation system")
st.image("image/capas/books-header.jpg")
st.header("Introduction")
st.write("Welcome to my recommendation system! I'm Paula Machado and this is part of my final project for Ironhack bootcamp on machine learning. This page allows you to find book recommendations based on a reference book of your choice using Goodreads users' information. You can find book options in the tab Books List to your left.")


df_model = extract.get_dataframe("model_prep")
df_retrieve = extract.get_dataframe("df_retrieve")
my_data = extract.get_dataframe("my_books")
my_prep_data = extract.get_dataframe("my_prep")

centroid_dict, cluster_dict = model.fit_Kmeans(df_model.values, 7)
# updating centroids
centroid_dict, cluster_dict = model.train(centroid_dict, cluster_dict, df_model.values)

book_ref = st.text_input("Please enter the index of the book you want to use as reference:")

if book_ref:
    # Convert user input to integer
    book_ref = int(book_ref)

    # Access the book reference
    reference_book = my_data.iloc[book_ref]

    # Get the title of the reference book
    book_title = reference_book['title']
    book_cover_filename = reference_book['cover_filename']

    # Construct the path to the book cover image file
    book_cover_path = "image/capas/" + book_cover_filename
    # Display the reference book details
    st.write("Reference Book:")
    st.write(book_title)
    st.image(book_cover_path, caption=book_title)

    # Access the new data point
    new_data_point = my_prep_data.iloc[book_ref].values.tolist()

    # Finding book recommendation
    nearest_point, second_nearest = model.predict_and_find_nearest(centroid_dict, cluster_dict, new_data_point)
    rec_link = df_retrieve.iloc[nearest_point]["link"]

    # Display the recommendation link
    st.write("Here is a link to the recommended book:")
    st.markdown(f"[{rec_link}]({rec_link})", unsafe_allow_html=True)

    
    # Open the first recommendation link automatically in a new tab
    if st.session_state.first_recommendation:
        components.html(f'<script>window.open("{rec_link}","_blank")</script>')
        st.session_state.first_recommendation = False

    # If the user didn't like the recommendation, show the second option
    if st.button("Didn't like the recommendation?"):
        second_rec_link = df_retrieve.iloc[second_nearest]["link"]
        st.write("Maybe you like this other:")
        st.markdown(f"[{second_rec_link}]({second_rec_link})", unsafe_allow_html=True)

# Initialize session state
if "first_recommendation" not in st.session_state:
    st.session_state.first_recommendation = True
        