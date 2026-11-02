# GeoLife data access

Dataset: Microsoft Research GeoLife GPS Trajectories 1.3 (182 user folders, 18,670 .plt files, ~24.9M points, 69 users with labels.txt).

- Not present on the machine initially (only an unrelated `Downloads/geolife_presentation.html`).
- Download succeeded on 2026-11-01 from the Microsoft download center (page id 52367, `download.microsoft.com/.../Geolife Trajectories 1.3.zip`, 313 MB, HTTP 200, no firewall block). Extracted unmodified to:

      geolife/inside/data/Geolife Trajectories 1.3/Data/<uid>/Trajectory/*.plt  (+ labels.txt for some users)

- `geolife/inside/data/` is git-ignored; raw data and the cached `points.parquet` are never committed.
- To refetch manually: open https://www.microsoft.com/en-us/download/details.aspx?id=52367, download the zip, extract into `geolife/inside/data/` so the path above exists.

Format notes: PLT has 6 header lines then `lat,lon,0,alt_ft,days_since_1899,date,time` (GMT). labels.txt is tab separated with a header (`Start Time  End Time  Transportation Mode`, times `YYYY/MM/DD HH:MM:SS`). The loader is `src/geolife/inside/data.py`; real user ids are replaced by seeded pseudonyms.
