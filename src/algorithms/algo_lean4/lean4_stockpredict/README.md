```
make clean && make all
make synth certify realism-audit 
make notebook
make notebook-synth

FLOW_ALLOW_DEV_RELEASE=1 make export
make bundle

pushd .
export DEPLOY_DIR=./deploy_dir
mkdir -p $DEPLOY_DIR && cp -R dist/bundle $DEPLOY_DIR && cd $DEPLOY_DIR/bundle
shasum -a 256 -c MANIFEST.sha256        # integrity check
./build_from_source.sh                  # elan/lake: re-check proofs, build generator
python3 python/distill.py               # release -> lean/out/genspec.json
./lean/.lake/build/bin/generate lean/out/genspec.json 8 168 out

less ./out/stock_synth.tsv
```
