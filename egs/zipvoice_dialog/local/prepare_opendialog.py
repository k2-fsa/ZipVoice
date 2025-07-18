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
This script prepares lhotse manifest files from the raw OpenDialog datasets.

We assume that you have downloaded the OpenDialog dataset and untarred the
tar files in audio/en and audio/zh so that the mp3 files are placed under
these two directories.

Download OpenDialog at https://huggingface.co/datasets/k2-fsa/OpenDialog
or https://www.modelscope.cn/datasets/k2-fsa/OpenDialog

"""

import argparse
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

from lhotse import CutSet, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.cut import Cut
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        help="The path of OpenDialog dataset.",
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

    recording_id = Path(wav_path).stem
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

    def _round_down(num, ndigits=0):
        factor = 10**ndigits
        return math.floor(num * factor) / factor

    uniq_id, text, wav_path, start, end = supervision
    try:
        recording_id = Path(wav_path).stem

        recording = recording_dict[recording_id]
        duration = (
            _round_down(end - start, ndigits=8)
            if end is not None
            else _round_down(recording.duration, ndigits=8)
        )
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
        logging.info(f"Error processing line: {e}")
        return None


def prepare_subset(
    jsonl_path: Pathlike,
    lang: str,
    sampling_rate: int,
    num_jobs: int,
    output_dir: Pathlike,
):
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param jsonl_path: Path to the jsonl file
    :param lang: Language of the subset
    :param sampling_rate: Target sampling rate of the audio
    :param num_jobs: Number of processes for parallel processing
    :param output_dir: Path where to write the manifests
    """
    logging.info(f"Preparing {lang} subset")

    # Step 1: Read all unique recording paths
    logging.info(f"Reading {jsonl_path}")
    recordings_path_set = set()
    supervision_list = list()
    with open(jsonl_path, "r") as fr:
        for line in fr:
            try:
                items = json.loads(line)
                uniq_id, text, wav_path = items["id"], items["text"], items["path"]
                start, end = 0, None
                recordings_path_set.add(jsonl_path.parent / wav_path)
                supervision_list.append((uniq_id, text, wav_path, start, end))
            except Exception as e:
                logging.warning(f"Error {e} when decoding JSON line: {line}")
                continue
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
    if sampling_rate != 24000:
        # All OpenDialog audios are 24kHz
        cut_set = cut_set.resample(sampling_rate)
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)

    logging.info("Saving cuts to disk...")
    # Step 5: Write manifests to disk
    cut_set.to_file(output_dir / f"opendialog_cuts_raw_{lang.upper()}-all.jsonl.gz")
    dev_cut_set = cut_set.subset(first=1000)
    dev_cut_set.to_file(output_dir / f"opendialog_cuts_raw_{lang.upper()}-dev.jsonl.gz")

    def remove_dev(c: Cut, set: set):
        if c.id in set:
            return False
        return True

    _remove_dev = partial(remove_dev, set=set(dev_cut_set.ids))
    train_cut_set = cut_set.filter(_remove_dev)
    train_cut_set.to_file(
        output_dir / f"opendialog_cuts_raw_{lang.upper()}-train.jsonl.gz"
    )


def prepare_dataset(
    dataset_path: Pathlike,
    sampling_rate: int,
    num_jobs: int,
    output_dir: Pathlike,
):
    for lang in ["en", "zh"]:
        jsonl_path = dataset_path / f"manifest.{lang}.jsonl"
        prepare_subset(
            jsonl_path=jsonl_path,
            lang=lang,
            sampling_rate=sampling_rate,
            num_jobs=num_jobs,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    args = get_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_dataset(
        dataset_path=dataset_path,
        sampling_rate=args.sampling_rate,
        num_jobs=args.num_jobs,
        output_dir=output_dir,
    )
