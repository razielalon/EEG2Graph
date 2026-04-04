from google.cloud import storage
import os

def download_from_gcs(bucket_name, prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for blob in bucket.list_blobs(prefix=prefix):
        local_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} -> {local_path}")

download_from_gcs("zuco_dataset_bucket", "", "./processed_zuco")