import os
import urllib

from zipfile import ZipFile


def download_data():
    """Download and extract the training data."""
    # download data
    data_file = "./data.zip"
    download_url = "https://azureopendatastorage.blob.core.windows.net/testpublic/temp/fowl_data.zip"
    urllib.request.urlretrieve(download_url, filename=data_file)

    # extract files
    with ZipFile(data_file, "r") as zip:
        print("extracting files...")
        zip.extractall()
        print("finished extracting")
        data_dir = zip.namelist()[0]

    # delete zip file
    os.remove(data_file)
    return data_dir


if __name__ == "__main__":
    data_dir = download_data()
    print(f"Downloaded sample data to {data_dir}")
