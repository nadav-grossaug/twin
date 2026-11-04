make clean
FLOW_ALLOW_DEV_RELEASE=1 make train
N_USERS=29641 make synth
pythonpp_flask_visualize_globe.py --left country_transfer.tsv --right synth/travel_synth.tsv
