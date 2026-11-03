-- DP input: dwell classes over fully observed visits (arrival AND departure
-- uncensored — a stay truncated by a recording break has unknown dwell).
{{ dp_hist(
    ['dw_lt1h', 'dw_1_3h', 'dw_3_8h', 'dw_8_14h', 'dw_ge14h'],
    "select user_id, dwell_class as cell from " ~ ref('feat_visits') ~ " where not arr_censored and not dep_censored",
    'dp_clip_visits'
) }}
