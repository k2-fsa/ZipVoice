#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script generates lhotse manifest files from TSV files for custom datasets.

Each line of the TSV files should be in one of the following formats:
1. "{uniq_id}\t{text}\t{wav_path}" if the text corresponds to the full wav",
2. "{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time} if text corresponds
    to part of the wav. The start_time and end_time specify the start and end
    times of the text within the wav, which should be in seconds.

Note: {uniq_id} must be unique for each line.

Usage:

Suppose you have two TSV files: "custom_train.tsv" and "custom_dev.tsv",
where "custom" is your dataset name, "train"/"dev" are used for training and
validation respectively.

(1) Prepare the training data

python3 -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/custom_train.tsv \
    --prefix "custom" \
    --subset "train" \
    --num-jobs 20 \
    --output-dir "data/manifests"

The output file would be "data/manifests/custom_cuts_train.jsonl.gz".

(2) Prepare the validation data

python3 -m zipvoice.bin.prepare_dataset \
    --tsv-path data/raw/custom_dev.tsv \
    --prefix "custom" \
    --subset "dev" \
    --num-jobs 1 \
    --output-dir "data/manifests"

The output file would be "data/manifests/custom_cuts_dev.jsonl.gz".

"""

import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from lhotse import CutSet, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tsv-path",
        type=str,
        help="The path of the tsv file. Each line should be in the format: "
        "{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time} "
        "if text corresponds to part of the wav or {uniq_id}\t{text}\t{wav_path} "
        "if the text corresponds to the full wav",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="custom",
        help="Prefix of the output manifest file.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="Subset name manifest file, typically train or dev.",
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=20,
        help="Number of jobs to processing.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="The destination directory of manifest files.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="The target sampling rate.",
    )
    return parser.parse_args()


def _parse_recording(
    wav_path: str,
) -> Tuple[Recording, str]:
    """
    :param wav_path: Path to the audio file
    :return: a tuple of "recording" and "recording_id"
    """

    recording_id = wav_path.replace("/", "_").replace(".", "_")
    recording = Recording.from_file(path=wav_path, recording_id=recording_id)

    return recording, recording_id


def _parse_supervision(
    supervision: List, recording_dict: dict
) -> Optional[SupervisionSegment]:
    """
    :param line: A line from the TSV file
    :param recording_dict: Dictionary mapping recording IDs to Recording objects
    :return: A SupervisionSegment object
    """

    uniq_id, text, wav_path, start, end = supervision
    try:
        recording_id = wav_path.replace("/", "_").replace(".", "_")

        recording = recording_dict[recording_id]
        duration = end - start if end is not None else recording.duration
        assert duration <= recording.duration, f"Duration {duration} is greater than "
        f"recording duration {recording.duration}"

        text = re.sub("_", " ", text)  # "_" is treated as padding symbol
        text = re.sub(r"\s+", " ", text)  # remove extra whitespace

        return SupervisionSegment(
            id=f"{uniq_id}",
            recording_id=recording.id,
            start=start,
            duration=duration,
            channel=recording.channel_ids,
            text=text.strip(),
        )
    except Exception as e:
        logging.warning(f"Error processing line: {e}")
        return None


def prepare_dataset(
    tsv_path: Pathlike,
    prefix: str,
    subset: str,
    sampling_rate: int,
    num_jobs: int,
    output_dir: Pathlike,
):
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param tsv_path: Path to the TSV file
    :param output_dir: Path where to write the manifests
    :param num_jobs: Number of processes for parallel processing
    :return: The CutSet containing the data
    """
    logging.info(f"Preparing {prefix} dataset {subset} subset.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{prefix}_cuts_{subset}.jsonl.gz"
    if (output_dir / file_name).is_file():
        logging.info(f"{file_name} exists, skipping.")
        return

    # Step 1: Read all unique recording paths
    recordings_path_set = set()
    supervision_list = list()
    with open(tsv_path, "r") as fr:
        for line in fr:
            items = line.strip().split("\t")
            if len(items) == 3:
                uniq_id, text, wav_path = items
                start, end = 0, None
            elif len(items) == 5:
                uniq_id, text, wav_path, start, end = items
                start, end = float(start), float(end)
            else:
                raise ValueError(
                    f"Invalid line format: {line},"
                    "requries to be 3 columns or 5 columns"
                )
            recordings_path_set.add(wav_path)
            supervision_list.append((uniq_id, text, wav_path, start, end))

    logging.info("Starting to process recordings...")
    # Step 2: Process recordings
    futures = []
    recording_dict = {}
    with ThreadPoolExecutor(max_workers=num_jobs) as ex:
        for wav_path in tqdm(recordings_path_set, desc="Submitting jobs"):
            futures.append(ex.submit(_parse_recording, wav_path))

        for future in tqdm(futures, desc="Processing recordings"):
            try:
                recording, recording_id = future.result()
                recording_dict[recording_id] = recording
            except Exception as e:
                logging.warning(
                    f"Error processing recording {recording_id} with error: {e}"
                )

        recording_set = RecordingSet.from_recordings(recording_dict.values())

    logging.info("Starting to process supervisions...")
    # Step 3: Process supervisions
    supervisions = []
    for supervision in tqdm(supervision_list, desc="Processing supervisions"):
        seg = _parse_supervision(supervision, recording_dict)
        if seg is not None:
            supervisions.append(seg)

    logging.info("Processing Cuts...")

    # Step 4: Create and validate manifests
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    cut_set = CutSet.from_manifests(
        recordings=recording_set, supervisions=supervision_set
    )
    cut_set = cut_set.sort_by_recording_id()
    cut_set = cut_set.resample(sampling_rate)
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)

    logging.info(f"Saving file to {output_dir / file_name}")
    # Step 5: Write manifests to disk
    cut_set.to_file(output_dir / file_name)
    logging.info("Done!")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()

    prepare_dataset(
        tsv_path=args.tsv_path,
        prefix=args.prefix,
        subset=args.subset,
        sampling_rate=args.sampling_rate,
        num_jobs=args.num_jobs,
        output_dir=args.output_dir,
    )
