#!/bin/bash
# Pack the results of finished runs for download (config, history, log, final.pt; no last.pt):
#   bash scripts/hpc/collect.sh            -> results_<time>.tar in the project folder
# Download it from the Open OnDemand file browser; on the Mac, `tar -xf results_*.tar` inside the repo.
cd "$(dirname "$0")/../.."
out=results_$(date +%m%d-%H%M).tar
files=$(ls runs/h_*/{config.json,history.json,train.log,final.pt,best.pt,exit_code} 2>/dev/null)
tar -cf "$out" $files
echo "$out: $(du -h "$out" | cut -f1)"
tar -tf "$out"
