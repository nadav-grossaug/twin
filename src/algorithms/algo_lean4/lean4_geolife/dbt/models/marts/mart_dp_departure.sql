-- DP input: departure-hour bins over visits with an observed departure.
{{ dp_hist(
    ['dep_night', 'dep_morning', 'dep_afternoon', 'dep_evening'],
    "select user_id, dep_bin as cell from " ~ ref('feat_visits') ~ " where not dep_censored",
    'dp_clip_visits'
) }}
