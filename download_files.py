"""
Last ned og pakk ut JetClass Pythia treningsdatasett (del 0)

Dette skriptet laster ned arkivet "JetClass_Pythia_train_100M_part0.tar" fra Zenodo,
validerer MD5-sjekksummen, og pakker ut innholdet direkte til prosjektets "Data"-mappe.

Arkivet inneholder en mappe med ROOT-filer.

Basert på: https://github.com/jet-universe/particle_transformer/blob/main/get_datasets.py

Bruk:
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
    Laster ned en fil fra en gitt URL med fremdriftsvisning.

    Parametre:
        url (str): Nettadressen filen skal lastes ned fra.
        dest_path (str): Full filsti hvor filen skal lagres.
        chunk_size (int): Antall byte per nedlastet blokk.

    Returnerer:
        str: Full sti til den nedlastede filen.
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
    Sjekker at filens hashverdi stemmer overens med forventet verdi.

    Parametre:
        file_path (str): Stien til filen som skal valideres.
        expected_hash (str): Forventet hashverdi (som streng).
        hash_alg (str): Hash-algoritme, enten 'md5' eller 'sha256'.
        chunk_size (int): Antall byte som leses per gang.

    Returnerer:
        bool: True hvis hash stemmer, ellers False.
    """
    if hash_alg.lower() == 'md5':
        hasher = hashlib.md5()
    elif hash_alg.lower() == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError("Støttet hash-algoritme er kun 'md5' eller 'sha256'.")

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


def extract_archive(archive_path, extract_to):
    """
    Pakker ut en tar-arkivfil til ønsket katalog.

    Parametre:
        archive_path (str): Full sti til tar-arkivet.
        extract_to (str): Katalogen arkivet skal pakkes ut til.

    Kaster:
        ValueError: Dersom filen ikke er et gyldig tar-arkiv.
    """
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, 'r') as tar:
            tar.extractall(path=extract_to)
    else:
        raise ValueError("Ugyldig arkivformat. Kun tar-arkiver støttes.")


def main():
    parser = argparse.ArgumentParser(
        description="Last ned og pakk ut JetClass Pythia treningsdatasett (del 0) i prosjektets Data-mappe."
    )
    parser.add_argument("--force", action="store_true", help="Tving ny nedlasting selv om filen allerede finnes.")
    args = parser.parse_args()

    # URL og forventet MD5-hash for del 0
    url = "https://zenodo.org/record/6619768/files/JetClass_Pythia_train_100M_part0.tar"
    expected_hash = "de4fd2dca2e68ab3c85d5cfd3bcc65c3"

    # Målmappe: Data-mappen i prosjektet
    project_dir = os.getcwd()
    data_dir = os.path.join(project_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "JetClass_Pythia_train_100M_part0.tar")

    # Last ned filen dersom den ikke finnes eller --force er spesifisert
    if os.path.exists(tar_path) and not args.force:
        print(f"Filen finnes allerede: {tar_path}. Validerer hash...")
        if validate_file(tar_path, expected_hash, hash_alg='md5'):
            print("Hash-verifisering vellykket. Hopper over nedlasting.")
        else:
            print("Hash-verifisering feilet. Laster ned på nytt.")
            os.remove(tar_path)
            download_file(url, tar_path)
    else:
        download_file(url, tar_path)

    # Verifiser at nedlastet fil er korrekt
    print("Validerer nedlastet fil...")
    if not validate_file(tar_path, expected_hash, hash_alg='md5'):
        raise RuntimeError("Filens hashverdi stemmer ikke med forventet verdi. Nedlastingen kan være korrupt.")

    # Pakk ut arkivet
    print("Pakker ut arkiv i Data-mappen...")
    extract_archive(tar_path, data_dir)
    print(f"Datasett pakket ut til {data_dir}")


if __name__ == "__main__":
    main()
