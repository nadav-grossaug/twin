you have a data file country_transfer.tsv, you need to build a tool chain that can generate differential-privacy synthethic data. you need to migrate the method that was used on another dataset. edit goal.md to reflect the plan for the data dataset. to do this, review the METHOD.md, review the existing goal.md that needs to be updated, and the llm wiki information. at all times You need to implement a persistent knowledge in LLM-Wiki in local filesystem as described in the link
  https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
  
implement the full duckdb+dbt, mini pytorch models, lean4 formulation, export lean4 from enclave, synth data generation, marimo notebooks. special notes regarding this dataset:
- focus on userid distribution of travels
- § models and sequence features
- retain the inter-travel date features, statistical moments, motifs
 you can review the existing implementation for the previus dataset in in ../lean4_stockpredict and re-use parts if relevant.

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
