# ReadME for AutoML Benchmark Template



## Install

1) Install venv for AutoML Benchmark (see AMLB tutorial) and activate the venv:
```bash
source ../venv/bin/activate
```

2Setup and env and test with (from root dir):
```bash
python runbenchmark.py AutoGluonLatest --userdir ./benchmarking-user-dir/
```

2) Correct and set up `submit-ag_ne.sh`
* Configure the SBATCH hardware settings
* Configure the output folder (#SBATCH -o)
* Configure the Environment Settings 
* Configure the Benchmark Settings
* Initialize the cache if needed (see ./download-benchmark-datasets.py)

3) Submit the job with:
```bash
sbatch --array=0-N%100 submit-ag_ne.sh
```

where N equals: `#folds * #tasks * #experiments`.

## Other

### Bump pyarrow 

In requirements.txt, bump the pyarrow version to pyarrow==17.0.0

### Searching for log files 
In the slurm output folder, you can search for log files with: `grep -m 1 -R . -e "wilt"`