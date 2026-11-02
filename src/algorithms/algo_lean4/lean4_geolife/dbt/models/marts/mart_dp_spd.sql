-- DP input: staypoints-per-day bins over user-days. Cell grid: the 5 PUBLIC
-- spd bins (1, 2, 3, 4-5, 6+).
{{ dp_hist(
    ['spd_1', 'spd_2', 'spd_3', 'spd_4_5', 'spd_6p'],
    "select user_id, spd_bin as cell from " ~ ref('feat_user_days'),
    'dp_clip_days'
) }}
