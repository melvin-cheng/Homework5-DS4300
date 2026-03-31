"""
Execute the DS4300 Cypher for Prof. Rachlin: seeds = Strokes/Regina songs in the graph,
one-hop SIMILAR_TO neighbors, exclude those artists, rank by max(edge score), top 5.

Requires: graph built by data_processing.py; NEO4J_* in .env. Run: `python recommendations.py`
"""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# max(r.score): best similarity to any seed song; LIMIT 5 = assignment requirement.
RECOMMEND_QUERY = """
MATCH (seed:Song)
WHERE toLower(seed.artists) CONTAINS 'the strokes'
   OR toLower(seed.artists) CONTAINS 'regina spektor'
WITH collect(DISTINCT seed) AS seeds
UNWIND seeds AS seed
MATCH (seed)-[r:SIMILAR_TO]->(rec:Song)
WHERE NOT (
  toLower(rec.artists) CONTAINS 'the strokes'
  OR toLower(rec.artists) CONTAINS 'regina spektor'
)
WITH rec, max(r.score) AS score
RETURN rec.artists AS artists,
       rec.album AS album,
       rec.name AS track,
       rec.genre AS genre,
       score
ORDER BY score DESC
LIMIT 5
"""

# SIMILAR_TO is stored twice per pair (bidirectional); /2 = unique undirected edges for reporting.
GRAPH_STATS_QUERY = """
MATCH (s:Song) WITH count(s) AS songCount
MATCH ()-[r:SIMILAR_TO]->() WITH songCount, count(r) AS relCount
RETURN songCount AS songs, relCount / 2 AS undirected_similar_pairs
"""


def main() -> None:
    if not NEO4J_PASSWORD:
        raise SystemExit("Set NEO4J_PASSWORD in .env or the environment.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        stats = session.run(GRAPH_STATS_QUERY).single()
        if stats:
            print(
                f"Graph: {stats['songs']} songs, "
                f"{stats['undirected_similar_pairs']} undirected similar pairs (SIMILAR_TO / 2)\n"
            )

        rows = session.run(RECOMMEND_QUERY)
        print("Top 5 recommendations (Artist | Album | Track | genre | score)\n")
        for i, record in enumerate(rows, start=1):
            print(
                f"{i}. {record['artists']} | {record['album']} | "
                f"{record['track']} | {record['genre']} | {record['score']:.6f}"
            )
    driver.close()


if __name__ == "__main__":
    main()
