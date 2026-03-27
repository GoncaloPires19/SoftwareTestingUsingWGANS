@echo off
set PYTHONIOENCODING=utf-8
python python/scripts/run_sf110_hypotheses.py --hypotheses H1 H1A H2 H3 --profile generalization-stress --epochs 150 --eval-samples 2000 --warmup-epochs 20 --skip-prepare --generate-synthetic --evaluate > training_out.log 2>&1
echo DONE >> training_out.log