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

Same-group edge share, observed vs chance (degree-weighted), all attributes:

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

## Caveat

Attribute fields, top-K values and homophily attributes were chosen from the real data. This selection is data-dependent and outside the DP guarantee in v1 (a DP top-K is v2); the released domains are treated as public.
