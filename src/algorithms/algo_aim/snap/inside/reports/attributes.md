# Attribute preparation (ego-Facebook)

Nodes: 4039, edges: 88234. Nodes with at least one feature: 99.8% (1406 distinct feature names merged from 10 ego networks).

Dropped identifier fields: first_name, middle_name, last_name, name.
Fields kept: birthday, education, gender, hometown, languages, locale, location, work. Fields not used: political, religion.

| attribute | feature prefix | domain size | coverage (not none) | other | none |
|---|---|---|---|---|---|
| gender | `gender` | 4 | 97.9% | 0.0% | 2.1% |
| locale | `locale` | 6 | 98.6% | 1.0% | 1.4% |
| birthday | `birthday` | 7 | 39.0% | 13.6% | 61.0% |
| school | `education;school;id` | 6 | 66.7% | 34.7% | 33.3% |
| employer | `work;employer;id` | 7 | 15.8% | 10.1% | 84.2% |
| location | `location;id` | 7 | 41.1% | 21.1% | 58.9% |
| hometown | `hometown;id` | 6 | 26.4% | 7.4% | 73.6% |
| language | `languages;id` | 6 | 18.5% | 1.5% | 81.5% |

## Homophily attributes

Groups used for `attr_edge_pair`: school (6 groups).

Same-group edge share, observed vs chance (degree-weighted), all attributes (counts missing-missing matches, so it UNDERSTATES homophily; the ranking below does not):

| attribute | observed | chance |
|---|---|---|
| gender | 0.545 | 0.503 |
| locale | 0.763 | 0.653 |
| birthday | 0.437 | 0.317 |
| school | 0.592 | 0.288 |
| employer | 0.723 | 0.700 |
| location | 0.406 | 0.323 |
| hometown | 0.539 | 0.425 |
| language | 0.670 | 0.658 |

## Candidate homophily groups (non-private ranking)

Lift per single attribute, over edges whose two ends are observed (neither endpoint has the attribute's missing label): same-value edge share divided by the degree-weighted chance level `sum_i a_i^2`. This is the data-dependent input for choosing `homophily_groups` in the config (outside the DP guarantee, see the caveat below).

| rank | attribute | lift | same-value share | chance | edges with both ends observed |
|---|---|---|---|---|---|
| 1 | hometown | 2.70 | 0.793 | 0.294 | 16.7% |
| 2 | location | 2.26 | 0.538 | 0.238 | 24.3% |
| 3 | school | 2.08 | 0.830 | 0.398 | 64.2% |
| 4 | birthday | 1.81 | 0.555 | 0.307 | 27.9% |
| 5 | employer | 1.37 | 0.645 | 0.472 | 4.2% |
| 6 | locale | 1.17 | 0.779 | 0.667 | 97.9% |
| 7 | gender | 1.08 | 0.563 | 0.520 | 96.9% |
| 8 | language | 1.04 | 0.633 | 0.607 | 4.9% |

## Caveat

Attribute fields, top-K values and homophily attributes were chosen from the real data. This selection is data-dependent and outside the DP guarantee in v1 (a DP top-K is v2); the released domains are treated as public.
