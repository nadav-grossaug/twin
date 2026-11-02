# SNAP data

Put one [Stanford SNAP](https://snap.stanford.edu/data/) edge list here, e.g.
`facebook_combined.txt.gz` (ego-Facebook) or `Wiki-Vote.txt.gz`. Files are used as-is
(`.txt` or `.txt.gz`); the loader reads directedness and sizes from the `#` header.

Add the file locally (it is not downloaded automatically). Set the `SNAP_FILE` environment variable
(file name inside this folder) to point the EDA notebook at a different graph.
