import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE


def matrix_corr(df, name):
    """
    This finction show the correlation matrix/heatmap for the input data.

    :arg:
        df: It takes a dataframe with the input data for the model
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().round(2)

    # Create the heatmap
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale='RdBu',
        showscale=True,
        reversescale=True)

    # Customize the layout
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis=dict(title='Features', side='bottom'),
        yaxis=dict(title='Features'),
        height=600,
        width=800,
        template='plotly_white')

    fig.write_image(f'image/plots/{name}.png')
    fig.write_html(f'image/plots/{name}.html')

    fig.show()
    

def elbow_method(df):
    """
    This function calculate the sum of squared distances for each k value and plots graph
    :arg:
        df: dataframe with all the data.
    :returns:
        The function does not have a return. It saves the plot inside the plots folder.
    """
    k_values = list(range(2, 11))
    sse = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)  
        sse.append(kmeans.inertia_)
    
    # Create the line chart
    fig = go.Figure(data=go.Scatter(x=list(k_values), y=sse, mode='lines+markers'))
    fig.update_layout(
        title='Elbow Method',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Sum of Squared Distances'),
        showlegend=False,
        template='plotly_white'
    )
    fig.write_image('image/plots/elbow_plot.png')
    fig.write_html('image/plots/elbow_plot.html')

    fig.show()

def silhouette_plot(df):
    """
    Generate an interactive line chart of Silhouette Coefficient vs. Number of Clusters using Plotly.

    :Args:
    df: (pandas DataFrame): The input data.
    """
    k_values = list(range(2, 11))
    silhouette_scores = []

    for k in k_values:
        # Perform clustering with K clusters
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(df)
        
        # Calculate the Silhouette Coefficient for the clustering result
        silhouette_coefficient = silhouette_score(df, cluster_labels)
        silhouette_scores.append(silhouette_coefficient)

    # Create the line chart
    fig = go.Figure(data=go.Scatter(x=k_values, y=silhouette_scores, mode='lines+markers'))
    fig.update_layout(
        title='Silhouette Coefficient vs. Number of Clusters',
        xaxis=dict(title='Number of Clusters (k)'),
        yaxis=dict(title='Silhouette Coefficient'),
        showlegend=False,
        template='plotly_white'
    )
    fig.write_image('image/plots/Silhouette_Coefficient.png')
    fig.write_html('image/plots/Silhouette_Coefficient.html')

    # Display the chart
    fig.show()

def davies_boulding_plot(df):
    """
    Generate an interactive line chart of Davies-Bouldin Index vs. Number of Clusters using Plotly.
    :Args:
    df (pandas DataFrame): The input data.
    """
    k_values = list(range(2, 11))
    davies_bouldin_scores = []

    for k in k_values:
        # Perform clustering with K clusters
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(df)
        
        # Calculate the Davies-Bouldin Index for the clustering result
        davies_bouldin_index = davies_bouldin_score(df, cluster_labels)
        davies_bouldin_scores.append(davies_bouldin_index)

    # Create the line chart
    fig = go.Figure(data=go.Scatter(x=list(k_values), y=davies_bouldin_scores, mode='lines+markers'))
    fig.update_layout(
        title='Davies-Bouldin Index vs. Number of Clusters',
        xaxis=dict(title='Number of Clusters (K)'),
        yaxis=dict(title='Davies-Bouldin Index'),
        showlegend=False,
        template='plotly_white'
    )
    fig.write_image('image/plots/Davies_Bouldin.png')
    fig.write_html('image/plots/Davies_Bouldin.html') 

    fig.show()


def kmeans_plot_blobs(df):

    tsne = TSNE(n_components=2, random_state=42)
    tsne_df = pd.DataFrame(tsne.fit_transform(df), columns = ['X', 'Y'] )

    sns.set(rc = {'figure.figsize':(15,8)})
    sns.scatterplot(x= 'X', y= 'Y',data= tsne_df).set(title= '2D Data Projection');

    # Plotting K-Means k=7
    km = KMeans(n_clusters = 7).fit(df)
    tsne_df['cluster'] = pd.Categorical(km.labels_)

    sns.set(rc = {'figure.figsize':(15,8)})
    sns.scatterplot(x='X', y='Y', hue='cluster', style='cluster', data= tsne_df).set(title= 'K-Means with k=7')