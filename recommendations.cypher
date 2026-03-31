// DS4300 — Five song recommendations (Prof. Rachlin: likes The Strokes + Regina Spektor)
//
// Model: (:Song)-[:SIMILAR_TO {score}]->(:Song); seed = songs whose artists match;
// score = max similarity to any seed; exclude Strokes / Regina Spektor tracks.
//
// Graph size: run in Python: python recommendations.py (prints counts + top 5)

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
LIMIT 5;
