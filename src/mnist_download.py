"""
Data acquisition layer for canonical MNIST IDX files.
Downloads the original four compressed IDX archives from the CVDFoundation mirror.
Author: Aryan Singh
"""
import os
import urllib.request
import hashlib
import ssl
import certifi
from datetime import datetime, timezone

CVDF_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

MNIST_FILES = {
    "train_images": {
        "filename": "train-images-idx3-ubyte.gz",
        "url": CVDF_BASE_URL + "train-images-idx3-ubyte.gz",
        "expected_sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "uncompressed_sha256": "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db",
        "uncompressed_bytes": 47040016,
        "description": "Training set images (60,000 samples, 28x28 pixels)"
    },
    "train_labels": {
        "filename": "train-labels-idx1-ubyte.gz",
        "url": CVDF_BASE_URL + "train-labels-idx1-ubyte.gz",
        "expected_sha256": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "uncompressed_sha256": "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5",
        "uncompressed_bytes": 60008,
        "description": "Training set labels (60,000 integer class IDs 0-9)"
    },
    "test_images": {
        "filename": "t10k-images-idx3-ubyte.gz",
        "url": CVDF_BASE_URL + "t10k-images-idx3-ubyte.gz",
        "expected_sha256": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "uncompressed_sha256": "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7",
        "uncompressed_bytes": 7840016,
        "description": "Test set images (10,000 samples, 28x28 pixels)"
    },
    "test_labels": {
        "filename": "t10k-labels-idx1-ubyte.gz",
        "url": CVDF_BASE_URL + "t10k-labels-idx1-ubyte.gz",
        "expected_sha256": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
        "uncompressed_sha256": "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2",
        "uncompressed_bytes": 10008,
        "description": "Test set labels (10,000 integer class IDs 0-9)"
    }
}

DATASET_PROVENANCE = {
    "dataset_name": "MNIST Handwritten Digit Database",
    "original_provenance_url": "https://yann.lecun.org/exdb/mnist/",
    "original_authors": "Yann LeCun, Corinna Cortes, Christopher J.C. Burges",
    "mirror_source": "CVDFoundation MNIST Mirror",
    "mirror_reference_url": "https://github.com/cvdfoundation/mnist",
    "secondary_reference_url": "https://huggingface.co/datasets/ylecun/mnist",
    "expected_train_count": 60000,
    "expected_test_count": 10000,
    "expected_image_shape": [28, 28],
    "expected_pixel_range": [0, 255],
    "expected_pixel_dtype": "uint8",
    "expected_label_range": [0, 9]
}


def compute_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_mnist_raw(target_dir="data/raw", force=False):
    """
    Downloads the 4 canonical MNIST compressed IDX files from CVDFoundation mirror.
    Verifies SHA-256 checksums upon download.
    """
    os.makedirs(target_dir, exist_ok=True)
    results = {}

    for key, info in MNIST_FILES.items():
        dest_path = os.path.join(target_dir, info["filename"])
        needs_download = force or not os.path.exists(dest_path)

        if needs_download:
            print(f"Downloading {info['filename']} from {info['url']}...")
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            req = urllib.request.Request(info["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_context) as response, open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    out_file.write(chunk)
            print(f"  ✓ Saved to {dest_path}")
        else:
            print(f"File {info['filename']} already exists locally.")

        file_size = os.path.getsize(dest_path)
        actual_sha256 = compute_sha256(dest_path)
        sha_match = (actual_sha256 == info["expected_sha256"])

        results[key] = {
            "filename": info["filename"],
            "path": dest_path,
            "size_bytes": file_size,
            "sha256": actual_sha256,
            "expected_sha256": info["expected_sha256"],
            "checksum_verified": sha_match
        }

        if not sha_match:
            print(f"  ⚠ WARNING: Checksum mismatch for {info['filename']}!")
        else:
            print(f"  ✓ Checksum verified (SHA-256: {actual_sha256[:12]}...)")

    return results
