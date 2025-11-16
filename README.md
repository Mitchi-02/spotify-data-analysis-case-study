# Spotify Data Mining Dashboard

An interactive web application built with Dash/Plotly to visualize and analyze Spotify dataset.

## Installation

### Prerequisites

-   Python 3.8 or higher
-   pip package manager

### Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Create a `data` folder and add the required CSV files:

```bash
mkdir -p data
```

Then place these files in the `data` folder: - `data/artists.csv` - `data/tracks.csv`

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Dash server:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://127.0.0.1:8050
```
