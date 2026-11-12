#!/bin/bash
# Queue state and the latest log line of every run. Paste the output back to the Mac side.
cd "$(dirname "$0")/../.."
squeue --me -o "%.10i %.18j %.9T %.10M %.10l %R"
for d in runs/h_*/; do
    r=$(basename "$d")
    code=$(cat "$d/exit_code" 2>/dev/null || echo "-")
    echo "[$r] exit=$code | $(tail -n 1 "$d/train.log" 2>/dev/null)"
done
