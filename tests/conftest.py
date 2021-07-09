from birds import download_data

def pytest_configure():
    download_data.download_data()
