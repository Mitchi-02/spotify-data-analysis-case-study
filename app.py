
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import data processing functions
from data_processing import load_and_process_data, get_audio_features

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Spotify Data Mining Dashboard"

# Load and process data
artists_df, tracks_df = load_and_process_data()
audio_features = get_audio_features()

# Sidebar layout
sidebar = html.Div(
    [
        html.H2("Spotify Analytics", className="display-6"),
        html.Hr(),
        html.P("Explore Spotify data mining insights", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Overview", href="/", active="exact"),
                dbc.NavLink("K-Means Clustering", href="/clustering", active="exact"),
                dbc.NavLink("DBSCAN Clustering", href="/dbscan", active="exact"),
                dbc.NavLink("Correlation Analysis", href="/correlation", active="exact"),
                dbc.NavLink("Genre & Features", href="/temporal", active="exact"),
                dbc.NavLink("Genre Analysis", href="/genres", active="exact"),
                dbc.NavLink("Top Charts", href="/top-charts", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    },
)

# Main content area with loading spinner
content = html.Div(
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="page-content"),
        fullscreen=True
    ),
    style={"margin-left": "18rem", "margin-right": "2rem", "padding": "2rem 1rem"}
)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    """Route to different pages based on URL"""
    if pathname == "/":
        return create_overview_page()
    elif pathname == "/clustering":
        return create_clustering_page()
    elif pathname == "/dbscan":
        return create_dbscan_page()
    elif pathname == "/correlation":
        return create_correlation_page()
    elif pathname == "/temporal":
        return create_temporal_page()
    elif pathname == "/genres":
        return create_genre_page()
    elif pathname == "/top-charts":
        return create_top_charts_page()
    return html.Div([
        html.H1("404: Not found", className="text-danger"),
        html.Hr(),
        html.P(f"The pathname {pathname} was not recognised..."),
    ])

# ============================================================================
# PAGE LAYOUTS
# ============================================================================

def create_overview_page():
    """Overview page with key statistics and distributions"""

    # Calculate key statistics
    total_artists = len(artists_df)
    total_tracks = len(tracks_df)
    avg_artist_popularity = artists_df['popularity'].mean()
    avg_track_popularity = tracks_df['popularity'].mean()

    # Create distribution plots
    artist_pop_fig = px.histogram(
        artists_df,
        x='popularity',
        nbins=50,
        title='Artist Popularity Distribution',
        labels={'popularity': 'Popularity', 'count': 'Frequency'},
        marginal='box'
    )

    track_pop_fig = px.histogram(
        tracks_df,
        x='popularity',
        nbins=50,
        title='Track Popularity Distribution',
        labels={'popularity': 'Popularity', 'count': 'Frequency'},
        marginal='box'
    )

    # Duration distribution (filter to < 10 minutes)
    duration_df = tracks_df[tracks_df['duration_ms'] < 600000].copy()
    duration_df['duration_min'] = duration_df['duration_ms'] / 60000

    duration_fig = px.histogram(
        duration_df,
        x='duration_min',
        nbins=50,
        title='Track Duration Distribution (< 10 minutes)',
        labels={'duration_min': 'Duration (minutes)', 'count': 'Frequency'}
    )

    return html.Div([
        html.H1("Spotify Data Mining Overview", className="mb-4"),

        # Key statistics cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_artists:,}", className="card-title text-primary"),
                        html.P("Total Artists", className="card-text"),
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_tracks:,}", className="card-title text-success"),
                        html.P("Total Tracks", className="card-text"),
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{avg_artist_popularity:.1f}", className="card-title text-info"),
                        html.P("Avg Artist Popularity", className="card-text"),
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{avg_track_popularity:.1f}", className="card-title text-warning"),
                        html.P("Avg Track Popularity", className="card-text"),
                    ])
                ])
            ], width=3),
        ], className="mb-4"),

        # Distribution plots
        dbc.Row([
            dbc.Col([dcc.Graph(figure=artist_pop_fig)], width=6),
            dbc.Col([dcc.Graph(figure=track_pop_fig)], width=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=duration_fig)], width=12),
        ]),
    ])


def create_clustering_page():
    """Clustering analysis page with K-Means and PCA visualization"""

    # Prepare data for clustering
    tracks_clean = tracks_df.dropna(subset=audio_features)

    # Sample data if too large (for performance)
    MAX_SAMPLES = 10000
    if len(tracks_clean) > MAX_SAMPLES:
        tracks_clean = tracks_clean.sample(n=MAX_SAMPLES, random_state=42)

    X = tracks_clean[audio_features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method data (use smaller n_init for speed)
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(
        x=list(K_range),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=10, color='blue'),
        line=dict(width=2)
    ))
    elbow_fig.update_layout(
        title='Elbow Method for Optimal K',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Inertia (Within-cluster sum of squares)',
        hovermode='x'
    )

    # Perform K-Means with optimal k=4
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=3)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Create PCA scatter plot
    track_name_col = 'track_name' if 'track_name' in tracks_clean.columns else 'name'
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'Popularity': tracks_clean['popularity'].values,
        'Track': tracks_clean[track_name_col].values if track_name_col in tracks_clean.columns else ['Track ' + str(i) for i in range(len(tracks_clean))]
    })

    pca_fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_data=['Track', 'Popularity'],
        title=f'K-Means Clustering Visualization (K={optimal_k}) - PCA Projection',
        labels={'Cluster': 'Cluster'},
        color_continuous_scale='viridis'
    )
    pca_fig.update_traces(marker=dict(size=5, opacity=0.6))

    # Cluster size analysis
    tracks_clean_copy = tracks_clean.copy()
    tracks_clean_copy['cluster'] = clusters
    cluster_sizes = tracks_clean_copy['cluster'].value_counts().sort_index().reset_index()
    cluster_sizes.columns = ['cluster', 'count']

    cluster_size_fig = px.bar(
        cluster_sizes,
        x='cluster',
        y='count',
        title='Number of Songs per Cluster',
        labels={'cluster': 'Cluster', 'count': 'Number of Songs'},
        color='count',
        color_continuous_scale='blues'
    )

    # Add sampling info if data was sampled
    sample_info = ""
    if len(tracks_df.dropna(subset=audio_features)) > MAX_SAMPLES:
        sample_info = f" (analyzing {MAX_SAMPLES:,} sampled tracks for performance)"

    return html.Div([
        html.H1("Clustering Analysis", className="mb-4"),
        html.P(f"K-Means clustering analysis of Spotify tracks based on audio features{sample_info}", className="lead"),

        # Elbow method
        dbc.Row([
            dbc.Col([dcc.Graph(figure=elbow_fig)], width=12),
        ], className="mb-4"),

        # PCA visualization
        dbc.Row([
            dbc.Col([dcc.Graph(figure=pca_fig)], width=12),
        ], className="mb-4"),

        # Cluster sizes
        dbc.Row([
            dbc.Col([dcc.Graph(figure=cluster_size_fig)], width=12),
        ], className="mb-4"),

        html.Hr(),
        html.H4("Cluster Interpretations", className="mt-4"),
        dbc.ListGroup([
            dbc.ListGroupItem([
                html.H5("Cluster 0: Calm, Instrumental, Acoustic"),
                html.P("Low energy, high instrumentalness, high acousticness - Relaxing background music"),
            ]),
            dbc.ListGroupItem([
                html.H5("Cluster 1: Energetic, Danceable"),
                html.P("High danceability, high energy - Dance/electronic oriented music"),
            ]),
            dbc.ListGroupItem([
                html.H5("Cluster 2: Acoustic, Vocal-heavy"),
                html.P("High acousticness, moderate energy - Singer-songwriter, acoustic pop"),
            ]),
            dbc.ListGroupItem([
                html.H5("Cluster 3: Electronic, High Energy"),
                html.P("Low acousticness, high energy, high instrumentalness - Electronic/synthetic music"),
            ]),
        ]),
    ])


def create_dbscan_page():
    """DBSCAN clustering analysis page"""

    # Prepare data for clustering
    tracks_clean = tracks_df.dropna(subset=audio_features)

    # Sample data if too large (for performance)
    MAX_SAMPLES = 10000
    if len(tracks_clean) > MAX_SAMPLES:
        tracks_clean = tracks_clean.sample(n=MAX_SAMPLES, random_state=42)

    X = tracks_clean[audio_features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN clustering with optimized parameters
    # eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
    # min_samples: minimum number of samples in a neighborhood for a point to be considered as a core point
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    clusters = dbscan.fit_predict(X_scaled)

    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Create PCA scatter plot
    track_name_col = 'track_name' if 'track_name' in tracks_clean.columns else 'name'
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters.astype(str),  # Convert to string for better color handling
        'Popularity': tracks_clean['popularity'].values,
        'Track': tracks_clean[track_name_col].values if track_name_col in tracks_clean.columns else ['Track ' + str(i) for i in range(len(tracks_clean))]
    })

    # Create a custom color map where -1 (noise) is gray
    pca_fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_data=['Track', 'Popularity'],
        title=f'DBSCAN Clustering Visualization - PCA Projection ({n_clusters} clusters, {n_noise} noise points)',
        labels={'Cluster': 'Cluster ID'}
    )
    pca_fig.update_traces(marker=dict(size=5, opacity=0.6))

    # Cluster statistics (excluding noise)
    tracks_clean_copy = tracks_clean.copy()
    tracks_clean_copy['cluster'] = clusters

    # Filter out noise points for statistics
    tracks_clustered = tracks_clean_copy[tracks_clean_copy['cluster'] != -1]

    if len(tracks_clustered) > 0:
        cluster_means = tracks_clustered.groupby('cluster')[audio_features].mean()

        # Cluster sizes
        cluster_sizes = tracks_clean_copy['cluster'].value_counts().reset_index()
        cluster_sizes.columns = ['Cluster', 'Size']
        cluster_sizes['Cluster'] = cluster_sizes['Cluster'].astype(str)

        size_fig = px.bar(
            cluster_sizes,
            x='Cluster',
            y='Size',
            title='Cluster Sizes',
            labels={'Cluster': 'Cluster ID', 'Size': 'Number of Tracks'},
            color='Size',
            color_continuous_scale='viridis'
        )
    else:
        size_fig = go.Figure()
        size_fig.update_layout(title='No clusters found')

    # Add sampling info if data was sampled
    sample_info = ""
    if len(tracks_df.dropna(subset=audio_features)) > MAX_SAMPLES:
        sample_info = f" (analyzing {MAX_SAMPLES:,} sampled tracks for performance)"

    return html.Div([
        html.H1("DBSCAN Clustering Analysis", className="mb-4"),
        html.P(f"Density-based clustering of Spotify tracks based on audio features{sample_info}", className="lead"),

        # Cluster statistics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{n_clusters}", className="card-title text-primary"),
                        html.P("Clusters Found", className="card-text"),
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{n_noise:,}", className="card-title text-warning"),
                        html.P("Noise Points", className="card-text"),
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{(n_noise/len(tracks_clean)*100):.1f}%", className="card-title text-info"),
                        html.P("Noise Percentage", className="card-text"),
                    ])
                ])
            ], width=4),
        ], className="mb-4"),

        # PCA visualization
        dbc.Row([
            dbc.Col([dcc.Graph(figure=pca_fig)], width=12),
        ], className="mb-4"),

        # Cluster sizes
        dbc.Row([
            dbc.Col([dcc.Graph(figure=size_fig)], width=12),
        ], className="mb-4"),

        html.Hr(),
        html.H4("DBSCAN Parameters", className="mt-4"),
        html.Ul([
            html.Li("eps=0.5: Maximum distance between two samples to be considered neighbors"),
            html.Li("min_samples=10: Minimum number of samples in a neighborhood to form a cluster"),
        ]),
    ])


def create_correlation_page():
    """Correlation analysis page"""

    # Calculate correlation matrix
    tracks_clean = tracks_df.dropna(subset=audio_features)
    corr_matrix = tracks_clean[audio_features].corr()

    # Create correlation heatmap
    corr_fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=audio_features,
        y=audio_features,
        title="Correlation Matrix of Audio Features",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect='auto'
    )
    corr_fig.update_traces(text=corr_matrix.round(2), texttemplate='%{text}')

    # Scatter plots for key correlations
    scatter1_fig = px.scatter(
        tracks_clean,
        x='danceability',
        y='popularity',
        title='Popularity vs Danceability',
        labels={'danceability': 'Danceability', 'popularity': 'Popularity'},
        trendline='ols',
        opacity=0.5
    )

    scatter2_fig = px.scatter(
        tracks_clean,
        x='energy',
        y='loudness',
        title='Energy vs Loudness',
        labels={'energy': 'Energy', 'loudness': 'Loudness (dB)'},
        trendline='ols',
        opacity=0.5,
        color='popularity',
        color_continuous_scale='viridis'
    )

    scatter3_fig = px.scatter(
        tracks_clean,
        x='acousticness',
        y='energy',
        title='Acousticness vs Energy',
        labels={'acousticness': 'Acousticness', 'energy': 'Energy'},
        trendline='ols',
        opacity=0.5
    )

    return html.Div([
        html.H1("Correlation Analysis", className="mb-4"),
        html.P("Explore relationships between audio features and popularity", className="lead"),

        # Correlation matrix
        dbc.Row([
            dbc.Col([dcc.Graph(figure=corr_fig)], width=12),
        ], className="mb-4"),

        # Scatter plots
        dbc.Row([
            dbc.Col([dcc.Graph(figure=scatter1_fig)], width=4),
            dbc.Col([dcc.Graph(figure=scatter2_fig)], width=4),
            dbc.Col([dcc.Graph(figure=scatter3_fig)], width=4),
        ]),
    ])


def create_temporal_page():
    """Temporal analysis page showing trends over time"""

    # Note: This dataset doesn't have release_year, so we'll analyze by genre and audio features instead
    # Create visualizations that make sense with available data

    # Audio features distribution by genre (top 10 genres)
    top_genres = tracks_df['genre'].value_counts().head(10).index if 'genre' in tracks_df.columns else []

    if len(top_genres) > 0:
        genre_filtered = tracks_df[tracks_df['genre'].isin(top_genres)]

        # Audio features by genre
        features_to_plot = ['danceability', 'energy', 'valence']
        genre_features = genre_filtered.groupby('genre')[features_to_plot].mean().reset_index()

        features_trend_fig = go.Figure()
        for feature in features_to_plot:
            features_trend_fig.add_trace(go.Bar(
                x=genre_features['genre'],
                y=genre_features[feature],
                name=feature.capitalize()
            ))
        features_trend_fig.update_layout(
            title='Average Audio Features by Top Genres',
            xaxis_title='Genre',
            yaxis_title='Average Value',
            barmode='group',
            xaxis_tickangle=-45
        )

        # Popularity by genre
        pop_by_genre = genre_filtered.groupby('genre')['popularity'].mean().sort_values(ascending=False).reset_index()
        pop_fig = px.bar(
            pop_by_genre,
            x='genre',
            y='popularity',
            title='Average Popularity by Genre',
            labels={'genre': 'Genre', 'popularity': 'Average Popularity'},
            color='popularity',
            color_continuous_scale='viridis'
        )
        pop_fig.update_xaxes(tickangle=-45)

        # Collaboration analysis by genre
        collab_by_genre = genre_filtered.groupby('genre')['num_artists'].mean().sort_values(ascending=False).reset_index()
        collab_fig = px.bar(
            collab_by_genre,
            x='genre',
            y='num_artists',
            title='Average Collaborations by Genre',
            labels={'genre': 'Genre', 'num_artists': 'Avg Artists per Track'},
            color='num_artists',
            color_continuous_scale='blues'
        )
        collab_fig.update_xaxes(tickangle=-45)
    else:
        # Fallback if no genre data
        features_trend_fig = go.Figure()
        features_trend_fig.update_layout(title='No genre data available')
        pop_fig = go.Figure()
        pop_fig.update_layout(title='No genre data available')
        collab_fig = go.Figure()
        collab_fig.update_layout(title='No genre data available')

    return html.Div([
        html.H1("Genre & Feature Analysis", className="mb-4"),
        html.P("Analyze how music characteristics vary across different genres", className="lead"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=pop_fig)], width=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=features_trend_fig)], width=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=collab_fig)], width=12),
        ]),
    ])


def create_genre_page():
    """Genre analysis page"""

    # Get genre distribution from tracks
    if 'genre' in tracks_df.columns:
        # Top genres overall
        top_genres = tracks_df['genre'].value_counts().head(20).reset_index()
        top_genres.columns = ['Genre', 'Count']

        genre_fig = px.bar(
            top_genres,
            x='Genre',
            y='Count',
            title='Top 20 Genres in Dataset',
            labels={'Genre': 'Genre', 'Count': 'Number of Tracks'},
            color='Count',
            color_continuous_scale='viridis'
        )
        genre_fig.update_xaxes(tickangle=-45)

        # Top genres in popular tracks (top 10%)
        threshold = tracks_df['popularity'].quantile(0.9)
        popular_tracks = tracks_df[tracks_df['popularity'] >= threshold]
        top_popular_genres = popular_tracks['genre'].value_counts().head(15).reset_index()
        top_popular_genres.columns = ['Genre', 'Count']

        popular_genre_fig = px.bar(
            top_popular_genres,
            x='Genre',
            y='Count',
            title='Top 15 Genres in Most Popular Tracks (Top 10%)',
            labels={'Genre': 'Genre', 'Count': 'Frequency'},
            color='Count',
            color_continuous_scale='plasma'
        )
        popular_genre_fig.update_xaxes(tickangle=-45)

        # Genre popularity distribution
        genre_pop = tracks_df.groupby('genre')['popularity'].mean().sort_values(ascending=False).head(15).reset_index()
        genre_pop_fig = px.bar(
            genre_pop,
            x='genre',
            y='popularity',
            title='Average Popularity by Genre (Top 15)',
            labels={'genre': 'Genre', 'popularity': 'Avg Popularity'},
            color='popularity',
            color_continuous_scale='blues'
        )
        genre_pop_fig.update_xaxes(tickangle=-45)
    else:
        # Fallback if no genre data
        genre_fig = go.Figure()
        genre_fig.update_layout(title='No genre data available')
        popular_genre_fig = go.Figure()
        popular_genre_fig.update_layout(title='No genre data available')
        genre_pop_fig = go.Figure()
        genre_pop_fig.update_layout(title='No genre data available')

    return html.Div([
        html.H1("Genre Analysis", className="mb-4"),
        html.P("Explore genre distribution and popularity across the dataset", className="lead"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=genre_fig)], width=12),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=popular_genre_fig)], width=6),
            dbc.Col([dcc.Graph(figure=genre_pop_fig)], width=6),
        ]),
    ])


def create_top_charts_page():
    """Top charts page with most popular artists and tracks"""

    # Top 10 artists
    if 'name' in artists_df.columns and 'popularity' in artists_df.columns:
        top_artists = artists_df.nlargest(10, 'popularity')[['name', 'popularity', 'streams']].copy()

        artists_fig = px.bar(
            top_artists,
            x='name',
            y='popularity',
            title='Top 10 Artists by Stream-based Popularity',
            labels={'name': 'Artist', 'popularity': 'Popularity Score'},
            color='popularity',
            color_continuous_scale='blues',
            hover_data=['streams']
        )
        artists_fig.update_xaxes(tickangle=-45)
    else:
        artists_fig = go.Figure()
        artists_fig.update_layout(title='Artist data not available')

    # Top 10 tracks
    if 'track_name' in tracks_df.columns:
        top_tracks = tracks_df.nlargest(10, 'popularity')[['track_name', 'artists', 'popularity']].copy()

        tracks_fig = px.bar(
            top_tracks,
            x='track_name',
            y='popularity',
            title='Top 10 Most Popular Tracks',
            labels={'track_name': 'Track', 'popularity': 'Popularity'},
            color='popularity',
            color_continuous_scale='greens',
            hover_data=['artists']
        )
        tracks_fig.update_xaxes(tickangle=-45)

        # Interactive data table for top tracks
        top_tracks_detailed = tracks_df.nlargest(20, 'popularity')[
            ['track_name', 'artists', 'popularity', 'danceability', 'energy', 'valence', 'genre']
        ].round(2)
    else:
        tracks_fig = go.Figure()
        tracks_fig.update_layout(title='Track data not available')
        top_tracks_detailed = pd.DataFrame()

    return html.Div([
        html.H1("Top Charts", className="mb-4"),
        html.P("Most popular artists and tracks in the dataset", className="lead"),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=artists_fig)], width=6),
            dbc.Col([dcc.Graph(figure=tracks_fig)], width=6),
        ], className="mb-4"),

        html.H3("Top 20 Tracks - Detailed View", className="mt-4 mb-3"),
        dbc.Table.from_dataframe(
            top_tracks_detailed,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size='sm'
        ) if not top_tracks_detailed.empty else html.P("No data available"),
    ])


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=8050)
