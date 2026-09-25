@echo off
if not exist raw_results.zip (
  echo ERROR: raw_results.zip is not included in the public package.
  echo Place an authorized copy in this folder, then run this script again.
  exit /b 1
)
python analyze_comporepair_stats.py raw_results.zip --output outputs --bootstrap 10000 --seed 20260830
pause
