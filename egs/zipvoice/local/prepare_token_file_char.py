#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang)
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

import argparse
import re
from collections import Counter
from pathlib import Path

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        type=Path,
        help="Path to the dict that maps the text tokens to IDs",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to the manifest file",
    )

    return parser.parse_args()


def prepare_tokens(manifest_file, token_file):
    counter = Counter()
    manifest = load_manifest_lazy(manifest_file)
    for cut in manifest:
        line = re.sub(r"\s+", " ", cut.supervisions[0].text)
        counter.update(line)

    unique_chars = set(counter.keys())

    if "_" in unique_chars:
        unique_chars.remove("_")

    sorted_chars = sorted(unique_chars, key=lambda char: counter[char], reverse=True)

    result = ["_"] + sorted_chars

    with open(token_file, "w", encoding="utf-8") as file:
        for index, char in enumerate(result):
            file.write(f"{char}\t{index}\n")


if __name__ == "__main__":
    args = get_args()
    prepare_tokens(args.manifest, args.tokens)
