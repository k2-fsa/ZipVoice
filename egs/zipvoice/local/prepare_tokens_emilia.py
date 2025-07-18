"""
This file reads the texts in given manifest and save the new cuts with phoneme tokens.
"""

import argparse
import glob
import logging
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path

from lhotse import load_manifest_lazy

from zipvoice.tokenizer.tokenizer import add_tokens


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        help="Subset of emilia, (ZH, EN, etc.)",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=50,
        help="Number of jobs to processing.",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/manifests/splits",
        help="The source directory of manifest files.",
    )

    parser.add_argument(
        "--dest-dir",
        type=str,
        help="The destination directory of manifest files.",
    )

    return parser.parse_args()


def prepare_tokens_emilia(file_name: str, input_dir: Path, output_dir: Path):
    logging.info(f"Processing {file_name}")
    if (output_dir / file_name).is_file():
        logging.info(f"{file_name} exists, skipping.")
        return

    try:
        cut_set = load_manifest_lazy(input_dir / file_name)
        cut_set = add_tokens(cut_set=cut_set, tokenizer="emilia")
        cut_set.to_file(output_dir / file_name)
    except Exception as e:
        logging.error(f"Manifest {file_name} failed with error: {e}")
        raise


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()

    input_dir = Path(args.source_dir)
    output_dir = Path(args.dest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cut_files = glob.glob(f"{args.source_dir}/emilia_cuts_{args.subset}.*.jsonl.gz")

    with Pool(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                prepare_tokens_emilia, filename.split("/")[-1], input_dir, output_dir
            )
            for filename in cut_files
        ]
        for f in futures:
            try:
                f.result()
                f.done()
            except Exception as e:
                logging.error(f"Future failed with error: {e}")
    logging.info("Processing done.")
