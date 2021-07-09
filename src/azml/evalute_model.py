import argparse


if __name__ == "__main__":
    # TODO implement this :)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="birds-evaluate")
    parser.add_argument("--dataset-version", type=str, default="latest")
    args = parser.parse_args()
