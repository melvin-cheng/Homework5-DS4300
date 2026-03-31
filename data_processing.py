"""
Load Spotify CSV, sample, compute pairwise cosine similarity on scaled audio features,
and build a Neo4j property graph for DS4300.

Graph: (:Song {…})-[:SIMILAR_TO {score}]->(:Song); (:Song)-[:PERFORMED_BY]->(:Artist).
Edges exist only when cosine similarity >= SIMILARITY_THRESHOLD (keeps the graph sparse).

Run: set NEO4J_* in .env, then `python data_processing.py`
"""
import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

SPOTIFY_CSV = os.getenv("SPOTIFY_CSV", "spotify.csv")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "2000"))

# Lower values add more SIMILAR_TO edges; higher = stricter “sounds alike” (often sparser graph).
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.99"))
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
 
# Spotify audio columns used as vectors for cosine similarity (after MinMax scaling).
FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
 
# Assignment test case: graph must include these artists’ tracks for Prof. Rachlin recommendations.
SEED_ARTISTS = ["The Strokes", "Regina Spektor"]

def load_and_sample(csv_path, sample_size):
    """All rows for SEED_ARTISTS plus a random sample of `sample_size` other tracks (random_state=42)."""
    df = pd.read_csv(csv_path)
 
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
 
    print(f"Total songs in dataset: {len(df)}")
 
    seed_mask = df["artists"].str.contains(
        "|".join(SEED_ARTISTS), case=False, na=False
    )
    seed_songs = df[seed_mask].copy()
    other_songs = df[~seed_mask].copy()
 
    print(f"\nSeed artist songs: {len(seed_songs)}")
    for artist in SEED_ARTISTS:
        count = df["artists"].str.contains(artist, case=False, na=False).sum()
        print(f"  {artist}: {count}")
 
    sample = other_songs.sample(n=min(sample_size, len(other_songs)), random_state=42)
    combined = pd.concat([seed_songs, sample], ignore_index=True)
    print(f"\nTotal songs in sample: {len(combined)}")
    print(f"  Genres represented: {combined['track_genre'].nunique()}")
 
    return combined

def normalize_features(df, feature_cols):
    """Scale features to [0,1] so loudness/tempo contribute fairly to cosine similarity."""
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def create_nodes(df):
    """MERGE each Song by Spotify track_id, then link to Artist nodes for optional graph queries."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        for _, row in df.iterrows():
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
    """Full pairwise cosine matrix on the sample; emit an edge per pair with score >= threshold."""
    features_matrix = df[feature_cols].values
    similarity_matrix = cosine_similarity(features_matrix)
    edges = []
    # Only j > i: each unordered pair once (matrix is symmetric).
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            score = similarity_matrix[i][j]
            if score >= threshold:
                edges.append((df.at[i, "track_id"], df.at[j, "track_id"], score))
    return edges

def create_edges_batch(session, edges, batch_size=500):
    # Undirected similarity: two directed edges so Cypher can traverse seed→rec in one hop.
    edge_records = [{"id_a": a, "id_b": b, "score": s} for a, b, s in edges]

    # Chunk UNWIND batches to avoid huge single transactions.
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
    # Full reset so re-runs don’t duplicate nodes/relationships.
    session.run("MATCH (n) DETACH DELETE n")
    print("Database cleared.")
 
 
def create_constraints(session):
    # Unique Song.id speeds up MATCH ... {id: $id} during edge loads.
    session.run(
        "CREATE CONSTRAINT song_id IF NOT EXISTS "
        "FOR (s:Song) REQUIRE s.id IS UNIQUE"
    )
    print("Constraint created on Song.id.")

def main():
    if not NEO4J_PASSWORD:
        raise SystemExit("Set NEO4J_PASSWORD in .env or the environment.")

    # 1) subset CSV  2) scale features  3) list similar pairs  4) wipe DB and reload nodes/edges
    df = load_and_sample(SPOTIFY_CSV, SAMPLE_SIZE)
    df = normalize_features(df, FEATURES)

    edges = compute_edges(df, FEATURES, SIMILARITY_THRESHOLD)
    print(f"Found {len(edges)} similar pairs above threshold {SIMILARITY_THRESHOLD}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        clear_database(session)
        create_constraints(session)
        create_nodes(df)
        create_edges_batch(session, edges)
    driver.close()
    

if __name__ == "__main__":
    main()