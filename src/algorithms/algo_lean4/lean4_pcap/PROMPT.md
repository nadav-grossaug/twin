❯ implement the tasks in goals.md, use dbt+duckdb, pytorch minimodels, lean4+mathlib to encode the motifs, stats and features of ./flow.jsonl
this data is a cisco-joy extracted features from pcap file used for cyber defense training, cisco joy: https://github.com/cisco/joy
make use of lean4 AWS contribution SampCert for formally verified probability distributions and random sampling . updated goal.md accordingly
add succint install and run instructions in HOWTO_run.md


❯ you have a data file flow.jsonl, you need to build a tool chain that can generate differential-privacy synthethic data. you need to migrate the method that was used on another dataset. edit goal.md to reflect the plan for the data dataset. to do this, review the METHOD.md, review the existing goal.md that needs to be updated, and the llm wiki information. at all times You need to implement a persistent knowledge in LLM-Wiki in local filesystem as described in the link
  https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
  
implement the full duckdb+dbt, mini pytorch models, lean4 formulation, export lean4 from enclave, synth data generation, marimo notebooks. special notes regarding this dataset:
 - this data is a cisco-joy extracted features from pcap file used for cyber defense training, cisco joy: https://github.com/cisco/joy
- add data-science analysis logic like chi2 statistical correlation analysis in the input data to find co-occurence and corellation betweenm columns, features and motif. 
- also look if there are sequences of input records that follow a timely flow logic.
you need to update the relevant dbt marts, lean4 formal description and pytorch minimodels accordingly


you can review the existing implementation for the previus dataset in in ../lean4_foursquare and re-use parts if relevant.

1. export LEAN_HOME=/opt/data/lean
change to always use install and reuse lean lake from LEAN_HOME

2. support only source only synth and deploy. discard bianry deploy. discard multiple bundling. only one bundling, support synth from bundle and synth from local dir

3. simplify the implementation: only the folling flows are interesting:

make clean : clean all data
make train : train the entire enclave flow including  duckdb+dbt, mini pytorch models, lean4 formulation, bundle build
make synth: synth local
make synth_from_bundle
make notebook-enclave
make notebook-synth

the only data used for synth is only lean4 files - no binary

update llm wiki, METHOD.md and goal.md accoring to the implementation progress so far
add $ make help -> to print summary of important hi-level make tasks

