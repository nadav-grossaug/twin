-- DP input: trip-displacement classes between consecutive staypoints
-- (visits with a predecessor).
{{ dp_hist(
    ['dist_lt1k', 'dist_1_3k', 'dist_3_10k', 'dist_ge10k'],
    "select user_id, dist_class as cell from " ~ ref('feat_visits') ~ " where dist_class is not null",
    'dp_clip_visits'
) }}
