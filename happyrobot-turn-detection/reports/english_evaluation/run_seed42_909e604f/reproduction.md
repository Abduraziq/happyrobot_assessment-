# Reproduction

Run from the project root in an environment with the serving requirements and `datasets` installed:

```powershell
python -m pip install -r requirements-serving.txt
python -m pip install datasets==5.1.0
python reports\english_evaluation\evaluate_english_cohort.py --output-dir reports\english_evaluation\reproduction_run
```

The script pins dataset revision `909e604f990651631a596f5cb03358b3ee8fbad0`, scans at most 20,000 streamed rows for metadata, excludes existing split IDs before sampling, uses seed `42`, and refuses to overwrite an existing output directory. It verifies the approved selected-ID hash before audio extraction. The script writes no files outside its selected output directory and checks hashes of the existing splits, model/config, and original test reports before and after execution.

The frozen model threshold is `0.3691340835714273`. The bootstrap confidence intervals use 10,000 stratified replicates and seed `20261107`. To reproduce in another fresh output location, choose a new directory under `reports\english_evaluation\`.
