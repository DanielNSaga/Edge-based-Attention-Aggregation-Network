"""
Download and extract the JetClass Pythia training dataset (part 0)

This script downloads the archive "JetClass_Pythia_train_100M_part0.tar" from Zenodo,
validates its MD5 checksum, and extracts the contents directly into the project's "Data" folder.

The archive contains a directory with ROOT files.

Based on: https://github.com/jet-universe/particle_transformer/blob/main/get_datasets.py

Usage:
    python download_files.py
"""

import argparse
import os
import tarfile
import hashlib
import requests
from tqdm import tqdm


def download_file(url, dest_path, chunk_size=1024):
    """
    Download a file from a given URL with a progress bar.

    Parameters:
        url (str): The URL to download the file from.
        dest_path (str): Full file path where the file will be saved.
        chunk_size (int): Number of bytes per downloaded chunk.

    Returns:
        str: Full path to the downloaded file.
    """
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    return dest_path


def validate_file(file_path, expected_hash, hash_alg='md5', chunk_size=8192):
    """
    Check that the file's hash matches the expected value.

    Parameters:
        file_path (str): Path to the file to validate.
        expected_hash (str): Expected hash value (as a string).
        hash_alg (str): Hash algorithm, either 'md5' or 'sha256'.
        chunk_size (int): Number of bytes to read per iteration.

    Returns:
        bool: True if the hash matches, otherwise False.
    """
    if hash_alg.lower() == 'md5':
        hasher = hashlib.md5()
    elif hash_alg.lower() == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError("Supported hash algorithms are only 'md5' or 'sha256'.")

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


def extract_archive(archive_path, extract_to):
    """
    Extract a tar archive to the desired directory.

    Parameters:
        archive_path (str): Full path to the tar archive.
        extract_to (str): Directory to extract the archive into.

    Raises:
        ValueError: If the file is not a valid tar archive.
    """
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(path=extract_to)
    else:
        raise ValueError("Invalid archive format. Only tar archives are supported.")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the JetClass Pythia training dataset (part 0) into the project's Data folder."
    )
    parser.add_argument("--force", action="store_true", help="Force re-download even if the file already exists.")
    args = parser.parse_args()

    # URL and expected MD5 hash for part 0
    url = "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar"
    expected_hash = "de4fd2dca2e68ab3c85d5cfd3bcc65c3"

    # Target folder: the project's Data directory
    project_dir = os.getcwd()
    data_dir = os.path.join(project_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "JetClass_Pythia_train_100M_part0.tar")

    # Download the file if it doesn't exist or --force is specified
    if os.path.exists(tar_path) and not args.force:
        print(f"File already exists: {tar_path}. Validating hash...")
        if validate_file(tar_path, expected_hash, hash_alg='md5'):
            print("Hash verification successful. Skipping download.")
        else:
            print("Hash verification failed. Re-downloading.")
            os.remove(tar_path)
            download_file(url, tar_path)
    else:
        download_file(url, tar_path)

    # Verify that the downloaded file is correct
    print("Validating downloaded file...")
    if not validate_file(tar_path, expected_hash, hash_alg='md5'):
        raise RuntimeError("The file's hash does not match the expected value. The download may be corrupted.")

    # Extract the archive
    print("Extracting archive into the Data folder...")
    extract_archive(tar_path, data_dir)
    print(f"Dataset extracted to {data_dir}")


if __name__ == "__main__":
    main()
