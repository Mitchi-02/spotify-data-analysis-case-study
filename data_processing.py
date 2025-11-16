"""
Data Processing Module
Handles loading and preprocessing of Spotify datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_process_data():
    """
    Load and preprocess artists and tracks datasets

    Returns:
        tuple: (artists_df, tracks_df)
    """
    import os

    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    # Load datasets from data folder
    # Note: This dataset has different structure - artists.csv is streaming stats
    # Most analysis will be done on tracks.csv which has the audio features
    artists_df = pd.read_csv(os.path.join(data_dir, 'artists.csv'), index_col=0)
    tracks_df = pd.read_csv(os.path.join(data_dir, 'tracks.csv'), index_col=0)

    # ========================================================================
    # ARTISTS DATA PREPROCESSING
    # ========================================================================

    # Clean column names
    artists_df.columns = artists_df.columns.str.strip()

    # Parse numeric columns (remove commas from numbers)
    if 'Lead Streams' in artists_df.columns:
        # Use pd.to_numeric with errors='coerce' to handle any conversion issues
        artists_df['Lead Streams'] = artists_df['Lead Streams'].astype(str).str.replace(',', '')
        artists_df['Lead Streams'] = pd.to_numeric(artists_df['Lead Streams'], errors='coerce')
    if 'Feats' in artists_df.columns:
        artists_df['Feats'] = artists_df['Feats'].astype(str).str.replace(',', '')
        artists_df['Feats'] = pd.to_numeric(artists_df['Feats'], errors='coerce')

    # Rename columns for consistency
    artists_df = artists_df.rename(columns={
        'Artist Name': 'name',
        'Lead Streams': 'streams',
        'Tracks': 'total_tracks',
        'One Billion': 'billion_streams_tracks',
        '100 Million': 'hundred_million_tracks'
    })

    # Create a popularity metric based on streams (normalized to 0-100)
    if 'streams' in artists_df.columns:
        max_streams = artists_df['streams'].max()
        artists_df['popularity'] = (artists_df['streams'] / max_streams * 100).round(2)

    # ========================================================================
    # TRACKS DATA PREPROCESSING
    # ========================================================================

    # Clean column names
    tracks_df.columns = tracks_df.columns.str.strip()

    # Handle missing values in audio features
    audio_features = get_audio_features()
    for feature in audio_features:
        if feature in tracks_df.columns:
            # Fill missing values with median
            tracks_df[feature] = pd.to_numeric(tracks_df[feature], errors='coerce')
            tracks_df[feature].fillna(tracks_df[feature].median(), inplace=True)

    # Count number of artists per track (by splitting the artists column)
    if 'artists' in tracks_df.columns:
        tracks_df['num_artists'] = tracks_df['artists'].str.count(';') + 1
        tracks_df['num_artists'].fillna(1, inplace=True)

    # Extract genre information
    if 'track_genre' in tracks_df.columns:
        tracks_df['genre'] = tracks_df['track_genre']

    # Ensure popularity is numeric
    if 'popularity' in tracks_df.columns:
        tracks_df['popularity'] = pd.to_numeric(tracks_df['popularity'], errors='coerce')
        tracks_df['popularity'].fillna(0, inplace=True)

    # Convert duration to minutes for easier interpretation
    if 'duration_ms' in tracks_df.columns:
        tracks_df['duration_min'] = tracks_df['duration_ms'] / 60000

    return artists_df, tracks_df


def get_audio_features():
    """
    Returns list of audio feature column names used in analysis

    Returns:
        list: Audio feature column names
    """
    return [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'valence',
        'tempo'
    ]


def get_top_popular_tracks(tracks_df, percentile=90):
    """
    Get top popular tracks based on percentile threshold

    Args:
        tracks_df: Tracks dataframe
        percentile: Percentile threshold (default 90 for top 10%)

    Returns:
        DataFrame: Filtered tracks above the popularity threshold
    """
    threshold = tracks_df['popularity'].quantile(percentile / 100)
    return tracks_df[tracks_df['popularity'] >= threshold]


def prepare_clustering_data(tracks_df):
    """
    Prepare data for clustering analysis

    Args:
        tracks_df: Tracks dataframe

    Returns:
        tuple: (features_matrix, clean_tracks_df)
    """
    from sklearn.preprocessing import StandardScaler

    audio_features = get_audio_features()

    # Remove rows with missing audio features
    tracks_clean = tracks_df.dropna(subset=audio_features)

    # Extract feature matrix
    X = tracks_clean[audio_features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, tracks_clean


def get_genre_statistics(tracks_df):
    """
    Calculate genre-related statistics

    Args:
        tracks_df: Tracks dataframe

    Returns:
        dict: Dictionary with genre statistics
    """
    from collections import Counter

    if 'genre' not in tracks_df.columns:
        return {
            'unique_genres': 0,
            'total_instances': 0,
            'genre_counts': Counter(),
            'top_10_genres': []
        }

    # Count all genres
    genre_counts = tracks_df['genre'].value_counts().to_dict()
    genre_counter = Counter(genre_counts)

    # Get unique genres
    unique_genres = tracks_df['genre'].nunique()
    total_genre_instances = len(tracks_df)

    return {
        'unique_genres': unique_genres,
        'total_instances': total_genre_instances,
        'genre_counts': Counter(genre_counts),
        'top_10_genres': tracks_df['genre'].value_counts().head(10).items()
    }
