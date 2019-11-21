import os
import sys
import argparse
import yaml

sys.path.append("..")
import utils.prefixer


def index_dataset(cfg, run_type):

    fetch_prefix = utils.prefixer.fetch_prefix(run_type)

    if os.path.exists(os.path.join(cfg["data"][fetch_prefix]["index_file"])):
        print("Index file exists. Not generating again...")
        return

    index_file = open(os.path.join(cfg["data"][fetch_prefix]["index_file"]), 'a')
    for r, d, f in os.walk(cfg["data"][fetch_prefix]["dataset_dir"]):
        for file in f:
            index_file.write(file+"\n")


def main():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="../configs/coco_unet.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    index_dataset(cfg, "train")
    index_dataset(cfg, "validate")
    index_dataset(cfg, "train")

if __name__ == "__main__":
    main()
