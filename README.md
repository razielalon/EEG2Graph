# EEG2Graph

Translating EEG brain signals into graph representations. 

## Setup:

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Getting the Data - processed or not:

The preprocessed ZuCo 2.0 `.mat` files are stored in a Google Cloud Storage bucket.

### Prerequisites

- Install and initialize the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- Authenticate with `gcloud auth login`
- Make sure you have read access to the `zuco_dataset_bucket` bucket

### Download

Run the download script:

```bash
python data_from_gcp.py
```

This downloads all files from the `zuco_dataset_bucket` bucket into `./processed_zuco/`.

## Preprocessing

If you dont have access to the bucket, and you have ZuCo 2.0 data as a .mat files, run the preprocessing pipeline:

```bash
python preprocessing/preprocess_zuco.py --data_dir /path/to/zuco2 --output_dir ./processed_zuco
```

You can inspect a `.mat` file before running the full pipeline:

```bash
python preprocessing/inspect_zuco.py /path/to/resultsYAC_NR.mat --detailed
```

### What the preprocessor does

1. Extracts word-level EEG frequency features (8 bands x 105 channels = 840 dims per word) from Tasks 1 (Normal Reading) and 2 (Task-Specific Reading)
2. Applies per-subject z-score normalization
3. Splits data into train/val/test sets (80/10/10), grouped by sentence to prevent leakage
4. Saves `.npy` and `.json` files to the output directory
