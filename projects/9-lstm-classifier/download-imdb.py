import os
import shutil
import sys
import tempfile
import urllib.request


IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
OUTPUT_NAME = "aclImdb"

def main():
    download_and_extract_archive()


def download_and_extract_archive():
    if os.path.exists(OUTPUT_NAME):
        print("Imdb dataset download target exists at " + OUTPUT_NAME)
    else:
        with urllib.request.urlopen(IMDB_URL) as response:
            with tempfile.NamedTemporaryFile() as temp_archive:
                temp_archive.write(response.read())
                imdb_tar = shutil.unpack_archive(
                    temp_archive.name, extract_dir=".", format="gztar")

    return


if __name__ == "__main__":
    sys.exit(main())