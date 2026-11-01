# Copyright 2026 ZipVoice contributors
# SPDX-License-Identifier: Apache-2.0
import time
import numpy as np


def encode_fixed(encoder, tokens, prompt_tokens, prompt_frames, total_frames):
    target = total_frames - prompt_frames
    names = [i.name for i in encoder.get_inputs()]
    speed = float(
        prompt_frames
        * (1 + np.count_nonzero(tokens) / np.count_nonzero(prompt_tokens))
        / (total_frames - 0.5)
    )
    lower = 0.0
    upper = float("inf")
    history = []
    started = time.perf_counter()
    for attempt in range(8):
        speed = float(np.float32(speed))
        result = encoder.run(
            None,
            {
                names[0]: tokens,
                names[1]: prompt_tokens,
                names[2]: np.array(prompt_frames, dtype=np.int64),
                names[3]: np.array(speed, dtype=np.float32),
            },
        )[0]
        frames = result.shape[1]
        history.append({"speed": speed, "target_frames": frames})
        if result.shape[1] == total_frames:
            return result, speed, time.perf_counter() - started, history
        # The encoder rounds its duration upward: (frames-1)*speed < C <= frames*speed.
        lower = max(lower, (frames - 1) * speed)
        upper = min(upper, frames * speed)
        speed = (lower + upper) * 0.5 / (total_frames - 0.5)
    raise RuntimeError("Cannot match engine length: " + str(history))
