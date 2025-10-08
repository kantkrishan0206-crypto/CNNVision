import os
import tarfile
import urllib.request

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = "data/raw"
ARCHIVE_PATH = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(ARCHIVE_PATH):
    print("Downloading CIFAR-10 dataset...")
    urllib.request.urlretrieve(URL, ARCHIVE_PATH)
    print("Download complete.")

    print("Extracting dataset...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print("Extraction complete.")
else:
    print("Dataset already exists at", ARCHIVE_PATH)
