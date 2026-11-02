# GeoLife EDA summary (from notebooks/01_eda.ipynb, real data)

- 182 users, 18,670 trajectories, 24,876,978 points; 2007-2012 (2008-09 = 79%). Heavy skew: top 10% of users hold 56% of points; median user 28 trajectories / 35k points; 10% of users have <720 points.
- Sampling: median dt 2 s, 96.5% <= 5 s; 2.8% of consecutive points have dt <= 0 (duplicates); 0.22% of steps imply > 70 m/s (GPS jumps); 1 invalid coordinate (lat 400).
- Trajectories: median 42 min / 10 km; max 97 h / 5,563 km (flights/trains).
- Space: 18.9% of points outside the Beijing box (101 of 182 users have some); 88.7% of in-Beijing points fall in a 0.4 x 0.3 degree core box (around the Haidian / 3rd-ring area).
- Labels: only 69 users, 14,718 segments, 21.8% of points inside a labelled span. Hours: walk 5.5k, bike 2.4k, car 1.7k, bus 1.5k, train 0.7k, taxi 0.7k, subway 0.4k.
- Stays (200 m / 20 min): 16,481 from 171 users; median 21 per user, p90 254, max 1,217 (very uneven); median duration 0.78 h.

## Users per H3 cell (the decisive number for user-level DP)

| res | kind | cells | cells with >=5 users | >=10 | >=20 | share of user-cell incidences in cells with >=10 users |
|---|---|---|---|---|---|---|
| 7 | points | 3,148 | 1,135 | 641 | 266 | 71% |
| 8 | points | 11,580 | 3,590 | 1,973 | 852 | 66% |
| 9 | points | 38,780 | 11,124 | 5,864 | 2,257 | 61% |
| 7 | stays | 552 | 124 | 67 | 25 | 59% |
| 8 | stays | 1,236 | 195 | 63 | 21 | 34% |
| 9 | stays | 2,575 | 196 | 58 | 10 | 18% |

Reading: transit/point-level overlap is strong (most traversed cells are shared by many users, i.e. road-network structure is learnable at user-level DP). Stay-level overlap is weak at res 8-9: only ~60 cells have >=10 distinct visiting users, and only 18-34% of stay incidences land in them. 'Common places' therefore carries signal mainly at coarse resolution (res 7, ~5 km2) or via category/land-use rather than exact place ids; with per-user contribution caps and noise sigma ~ cap, expect only the top tens of cells (campus/centre hubs) to clear the noise at res 8-9.

Caveats: the pseudonym mapping is seeded but the notebook output contains no real user ids; stay detection ran on points thinned to 30 s and speed-filtered; per-user cell sets are not capped here (a DP pipeline will cap contributions, which lowers these counts for heavy users).
