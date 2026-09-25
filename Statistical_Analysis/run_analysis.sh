#!/bin/sh
set -eu
if [ ! -f raw_results.zip ]; then
  echo "ERROR: raw_results.zip is not included in the public package." >&2
  echo "Place an authorized copy in this directory, then run this script again." >&2
  exit 1
fi
python3 analyze_comporepair_stats.py raw_results.zip --output outputs --bootstrap 10000 --seed 20260830
