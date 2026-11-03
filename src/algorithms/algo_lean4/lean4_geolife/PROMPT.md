❯ you have a data file geolife_data.tsv (Microsoft GeoLife GPS trajectories,
labelled subset), you need to build a tool chain that can generate
differential-privacy synthetic data. you need to migrate the method that was
used on another dataset. edit goal.md to reflect the plan for the dataset. to
do this, review the METHOD.md, review the existing goal.md that needs to be
updated, and the llm wiki information. at all times You need to implement a
persistent knowledge in LLM-Wiki in local filesystem as described in the link
  https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

implement the full duckdb+dbt, mini pytorch models, lean4 formulation, export
lean4 from enclave, synth data generation, marimo notebooks. special notes
regarding this dataset:
- add data-science analysis logic like chi2 statistical correlation analysis
  in the input data to find co-occurence and correlation between columns,
  features and motifs.
- also look if there are sequences of input records that follow a timely flow
  logic.
- you need to update the relevant dbt marts, lean4 formal description and
  pytorch minimodels accordingly
- trajectories should include noise, estimate the statistical moments of noise
- trajectories should not follow straight lines and strict 90 angle turns
- users' pattern of daily behaviour: the motifs should follow staypoints and
  anchors behaviour of users' pattern of daily behaviour
- add a manipulation step in the synth, where users should be juxtaposed from
  beijing to tokyo. do not include beijing lat/long coords in the synth data

you can review the existing implementation for the previous datasets in
../lean4_foursquare (port base) and ../lean4_pcap, and re-use parts if
relevant.

1. export LEAN_HOME=/opt/data/lean
   change to always use install and reuse lean lake from LEAN_HOME

2. support only source only synth and deploy. discard binary deploy. discard
   multiple bundling. only one bundling, support synth from bundle and synth
   from local dir

3. simplify the implementation: only the following flows are interesting:

   make clean : clean all data
   make train : train the entire enclave flow including duckdb+dbt, mini
                pytorch models, lean4 formulation, bundle build
   make synth : synth local
   make synth_from_bundle
   make notebook-enclave
   make notebook-synth

the only data used for synth is only lean4 files - no binary

update llm wiki, METHOD.md and goal.md according to the implementation
progress so far. add $ make help -> to print summary of important hi-level
make tasks

(Note: the original pasted brief said "flow.jsonl ... cisco-joy extracted
features" in two places — that text was carried over from ../lean4_pcap's
PROMPT.md; the dataset-specific requirements (staypoints/anchors, Beijing →
Tokyo juxtaposition, GPS noise, no straight lines) and this repo's data file
identify the actual target as the GeoLife trajectories above.)
