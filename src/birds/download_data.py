import os
import urllib.request

from zipfile import ZipFile


def strip_top_level(file_name, top_level="fowl_data/"):
    return file_name[len(top_level):] if top_level in file_name else file_name


def download_data(output_dir="./data"):
    """Download and extract the training data."""
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        print("Already exists")
        return output_dir
    
    # download data
    download_url = "https://azureopendatastorage.blob.core.windows.net/testpublic/temp/fowl_data.zip"
    data_file, _ = urllib.request.urlretrieve(download_url)

    os.makedirs(output_dir, exist_ok=True)

    # extract files
    with ZipFile(data_file, "r") as zip_file:
        print("Extracting files...")
        for zip_info in zip_file.infolist():
            original = zip_info.filename
            stripped = strip_top_level(original)
            if stripped:
                zip_info.filename = stripped
                zip_file.extract(zip_info, path=output_dir)
        print("Finished extracting")

    # delete zip file
    os.remove(data_file)
    return output_dir


if __name__ == "__main__":
    data_dir = download_data()
    print(f"Sample data in {data_dir}")
