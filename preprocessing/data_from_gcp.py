"""
Download preprocessed ZuCo data from the GCS bucket.

The bucket `zuco_dataset_bucket` is organized as:
    ZuCo1/ ...  (ZuCo 1.0 processed artifacts)
    ZuCo2/ ...  (ZuCo 2.0 processed artifacts)

Usage:
    python data_from_gcp.py --dataset zuco2
    python data_from_gcp.py --dataset zuco1 --output_dir ./processed_zuco1
"""

import argparse
import os
from google.cloud import storage


BUCKET_NAME = "zuco_dataset_bucket"

DATASET_PREFIXES = {
    "zuco1": "ZuCo1/",
    "zuco2": "ZuCo2/",
}

DEFAULT_OUTPUT_DIRS = {
    "zuco1": "./processed_zuco1",
    "zuco2": "./processed_zuco",
}


def download_from_gcs(bucket_name, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for blob in bucket.list_blobs(prefix=prefix):
        # Skip "folder placeholder" blobs that end with a slash
        if blob.name.endswith("/"):
            continue
        local_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} -> {local_path}")


def main():
    parser = argparse.ArgumentParser(description="Download processed ZuCo data from GCS")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_PREFIXES.keys()),
        help="Which dataset folder to fetch from the bucket",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local directory to download into (default: ./processed_zuco for zuco2, ./processed_zuco1 for zuco1)",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=BUCKET_NAME,
        help=f"GCS bucket name (default: {BUCKET_NAME})",
    )
    args = parser.parse_args()

    prefix = DATASET_PREFIXES[args.dataset]
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[args.dataset]

    print(f"Dataset:     {args.dataset}")
    print(f"Bucket:      {args.bucket}")
    print(f"Prefix:      {prefix}")
    print(f"Local dir:   {output_dir}\n")

    download_from_gcs(args.bucket, prefix, output_dir)


if __name__ == "__main__":
    main()
