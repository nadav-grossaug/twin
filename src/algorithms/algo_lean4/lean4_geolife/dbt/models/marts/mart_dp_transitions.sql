-- DP input: anchor-role transitions (3x3 grid home/work/other, row-major).
-- Only seq_ok visits count (gap <= visit_gap_break_s, same local day — no
-- false transitions over recording gaps; a user's first visit never counts).
{{ dp_hist(
    ['home__home', 'home__work', 'home__other',
     'work__home', 'work__work', 'work__other',
     'other__home', 'other__work', 'other__other'],
    "select user_id, trans_cell as cell from " ~ ref('feat_visits') ~ " where seq_ok",
    'dp_clip_visits'
) }}
