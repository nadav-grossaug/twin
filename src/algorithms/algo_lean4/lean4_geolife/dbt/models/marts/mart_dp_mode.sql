-- DP input: mode-class mix over labelled legs. 'XX' (off-map labels) is
-- filtered here and tallied in q_data_integrity instead.
{{ dp_hist(
    ['walk', 'bike', 'bus', 'car', 'rail', 'other'],
    "select user_id, mode_class as cell from " ~ ref('feat_legs') ~ " where mode_class != 'XX'",
    'dp_clip_legs'
) }}
