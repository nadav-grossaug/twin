-- DP input: daily-routine motif classes over user-days (the behavioral
-- backbone; the generation unit). Cell grid: the 5 PUBLIC motif classes.
{{ dp_hist(
    ['home_day', 'commute', 'errand_loop', 'tour', 'irregular'],
    "select user_id, motif_class as cell from " ~ ref('feat_user_days'),
    'dp_clip_days'
) }}
