import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

SPOTIFY_CSV = os.getenv("SPOTIFY_CSV", "spotify.csv")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "2000"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.99"))
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
 
# Audio features
FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
 
SEED_ARTISTS = ["The Strokes", "Regina Spektor"]

def load_and_sample(csv_path, sample_size):
    df = pd.read_csv(csv_path)
 
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
 
    print(f"Total songs in dataset: {len(df)}")
 
    # Separate seed artist songs
    seed_mask = df["artists"].str.contains(
        "|".join(SEED_ARTISTS), case=False, na=False
    )
    seed_songs = df[seed_mask].copy()
    other_songs = df[~seed_mask].copy()
 
    print(f"\nSeed artist songs: {len(seed_songs)}")
    for artist in SEED_ARTISTS:
        count = df["artists"].str.contains(artist, case=False, na=False).sum()
        print(f"  {artist}: {count}")
 
    # Random sample from the rest
    sample = other_songs.sample(n=min(sample_size, len(other_songs)), random_state=42)
 
    # Combine and reset index
    combined = pd.concat([seed_songs, sample], ignore_index=True)
    print(f"\nTotal songs in sample: {len(combined)}")
    print(f"  Genres represented: {combined['track_genre'].nunique()}")
 
    return combined

def normalize_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def create_nodes(df):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for _, row in df.iterrows():
            # Create song node
            session.run(
                """
                MERGE (s:Song {id: $id})
                SET s.name = $name,
                    s.artists = $artists,
                    s.album = $album,
                    s.genre = $genre,
                    s.danceability = $danceability,
                    s.energy = $energy,
                    s.loudness = $loudness,
                    s.speechiness = $speechiness,
                    s.acousticness = $acousticness,
                    s.instrumentalness = $instrumentalness,
                    s.liveness = $liveness,
                    s.valence = $valence,
                    s.tempo = $tempo
                """,
                id=row["track_id"],
                name=row["track_name"],
                artists=row["artists"],
                album=row["album_name"],
                genre=row["track_genre"],
                danceability=row["danceability"],
                energy=row["energy"],
                loudness=row["loudness"],
                speechiness=row["speechiness"],
                acousticness=row["acousticness"],
                instrumentalness=row["instrumentalness"],
                liveness=row["liveness"],
                valence=row["valence"],
                tempo=row["tempo"]
            )
            # Create artist nodes and relationships
            for artist in row["artists"].split(","):
                artist_name = artist.strip()
                session.run(
                    """
                    MERGE (a:Artist {name: $name})
                    """,
                    name=artist_name
                )
                session.run(
                    """
                    MATCH (s:Song {id: $song_id}), (a:Artist {name: $artist_name})
                    MERGE (s)-[:PERFORMED_BY]->(a)
                    """,
                    song_id=row["track_id"],
                    artist_name=artist_name
                )
    driver.close()


def compute_edges(df, feature_cols, threshold):
    features_matrix = df[feature_cols].values
    similarity_matrix = cosine_similarity(features_matrix)
    edges = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            score = similarity_matrix[i][j]
            if score >= threshold:
                edges.append((df.at[i, "track_id"], df.at[j, "track_id"], score))
    return edges

def create_edges_batch(session, edges, batch_size=500):
    edge_records = [{"id_a": a, "id_b": b, "score": s} for a, b, s in edges]

    for start in range(0, len(edge_records), batch_size):
        batch = edge_records[start:start + batch_size]
        session.run("""
            UNWIND $batch AS edge
            MATCH (a:Song {id: edge.id_a}), (b:Song {id: edge.id_b})
            MERGE (a)-[:SIMILAR_TO {score: edge.score}]->(b)
            MERGE (b)-[:SIMILAR_TO {score: edge.score}]->(a)
        """, {"batch": batch})

    print(f"Created {len(edges) * 2} SIMILAR_TO relationships (bidirectional).")

def clear_database(session):
    session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")
 
 
def create_constraints(session):
    session.run(
        "CREATE CONSTRAINT song_id IF NOT EXISTS "
        "FOR (s:Song) REQUIRE s.id IS UNIQUE"
    )
    print("Constraint created on Song.id.")

def main():
    if not NEO4J_PASSWORD:
        raise SystemExit("Set NEO4J_PASSWORD in .env or the environment.")

    # Load and preprocess data
    df = load_and_sample(SPOTIFY_CSV, SAMPLE_SIZE)
    df = normalize_features(df, FEATURES)

    # Compute similar pairs before opening DB connection
    edges = compute_edges(df, FEATURES, SIMILARITY_THRESHOLD)
    print(f"Found {len(edges)} similar pairs above threshold {SIMILARITY_THRESHOLD}")

    # Create nodes and edges in Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        clear_database(session)
        create_constraints(session)
        create_nodes(df)
        create_edges_batch(session, edges)
    driver.close()
    

if __name__ == "__main__":
    main()