import streamlit as st
import streamlit.components.v1 as components
import src.extraction as extract


st.header("List of books")
st.write("Here are some books with prepared information to get new recommendation. Please use the index when inputing the chosen book.")

my_data = extract.get_dataframe("my_books")

st.table(my_data[["title","subgenre","description"]])
