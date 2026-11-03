
mkdir -p ~/workdisk/git/twin/src/algorithms/algo_lean4/lean4_booking
cd      ~/workdisk/git/twin/src/algorithms/algo_lean4/lean4_booking
cp -R ../lean4_geolife/METHOD.md ../lean4_geolife/goal.md ../lean4_geolife/HOWTO_run.md ../lean4_geolife/wiki .


goal: merge and update local folder llm wiki 
description: 
we have a local implementation of  a persistent knowledge in LLM-Wiki in local filesystem as described in the link  https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
  
each folder has its own copy of wiki in ./wiki. You need to update this folders
  ../wiki based on aggregation of all data in all ../*/wiki
$ du -hs ../lean4_booking/wiki ../lean4_geolife/wiki ../lean4_snap/wiki ../lean4_foursquare/wiki ../lean4_pcap/wiki ../lean4_stockpredict/wiki
252K    ../lean4_booking/wiki
252K    ../lean4_geolife/wiki
288K    ../lean4_snap/wiki
208K    ../lean4_foursquare/wiki
224K    ../lean4_pcap/wiki
176K    ../lean4_stockpredict/wiki
you need to update ./wiki accoridingly

❯ implement the tasks in goals.md, use dbt+duckdb, pytorch minimodels, lean4+mathlib to encode the motifs, stats and features of ```./combined_set.csv```
this data is ```a millions of booking.com real anonymized accommodation reservations to come up with a strategy for making the best recommendation for the next destination in real-time.```

make use of lean4 AWS contribution SampCert for formally verified probability distributions and random sampling . updated goal.md accordingly
add succint install and run instructions in HOWTO_run.md

for important insights about the dataset and goals, deep think about the information in ```README.md```

❯ you have a data file  ```./combined_set.csv``` in csv format, you need to build a tool chain that can generate differential-privacy synthethic data. you need to migrate the method that was used on another dataset. edit goal.md to reflect the plan for the data dataset. to do this, review the METHOD.md, review the existing goal.md that needs to be updated, and the llm wiki information. at all times You need to implement a persistent knowledge in LLM-Wiki in local filesystem as described in the link
  https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
  
implement the full duckdb+dbt, mini pytorch models, lean4 formulation, export lean4 from enclave, synth data generation, marimo notebooks. special notes regarding this dataset:
- add data-science analysis logic like chi2 statistical correlation analysis in the input data to find co-occurence and corellation betweenm columns, features and motif. 
- also look if there are sequences of input records that follow a timely flow logic.
you need to update the relevant dbt marts, lean4 formal description and pytorch minimodels accordingly
-  The SNAP Facebook dataset is particularly useful because it has strong community structure and a published average clustering coefficient of approximately 0.6055.

consider these  learning targets:


you can review the existing implementation for the previus dataset in in ../lean4_foursquare and ../lean4_geolife and re-use parts if relevant.

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


# 
ε reallocation: improve synth data fidelty even if privacy bounds are worse off. 

check number of records and user, and synth distribution of number of records
  correlates the original enclave data

are there other known unaddressed issues? rank by importance

evaluate notebook-synth for leads for optimization.
are there other known unaddressed issues? rank by importance

anohter pass: attempt to add additional pytorch minimodels, motifs, features to improve synth fidelity


algo_lean4/lean4_geolife lgo_lean4/lean4_snap
