from typing import List

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence

punctuation = {";", ":", ",", ".", "!", "?", "；", "：", "，", "。", "！", "？"}


def chunk_tokens_punctuation(tokens_list: List[str], max_tokens: int = 100):
    """
    Splits the input tokens list into chunks according to punctuations,
        each with a maximum number of tokens.

    Args:
        token_list (list of str): The list of tokens to be split.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        List[str]: A list of text chunks.
    """

    # 1. Split the tokens according to punctuations.
    sentences = []
    current_sentence = []
    for token in tokens_list:
        # If the first token of current sentence is punctuation or blank,
        # append it to the end of the previous sentence.
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in punctuation or token == " ")
        ):
            sentences[-1].append(token)
        # Otherwise, append the current token to the current sentence.
        else:
            current_sentence.append(token)
            # Split the sentence in positions of punctuations.
            if token in punctuation:
                sentences.append(current_sentence)
                current_sentence = []
    # Assume the last few tokens are also a sentence
    if len(current_sentence) != 0:
        sentences.append(current_sentence)

    # 2. Merge short sentences.
    chunks = []
    current_chunk = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk.extend(sentence)
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = sentence

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


def chunk_tokens_dialog(tokens_list: List[str], max_tokens: int = 100):
    """
    Splits the input tokens list into chunks according to speaker-turn
        symbol [S1], each with a maximum number of tokens.

    Args:
        token_list (list of str): The list of tokens to be split.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        List[str]: A list of text chunks.
    """

    # 1. Split the tokens according to speaker-turn symbol [S1].
    dialogs = []
    current_dialog = []
    for token in tokens_list:
        if token == "[S1]":
            if len(current_dialog) != 0:
                dialogs.append(current_dialog)
            current_dialog = []
        current_dialog.append(token)
    # Assume the last few tokens are also a dialog
    if len(current_dialog) != 0:
        dialogs.append(current_dialog)

    # 2. Merge short dialogs.
    chunks = []
    current_chunk = []
    for dialog in dialogs:
        if len(current_chunk) + len(dialog) <= max_tokens:
            current_chunk.extend(dialog)
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = dialog

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


def batchify_tokens(
    tokens_list: List[List[int]],
    max_duration: float,
    prompt_duration: float,
    token_duration: float,
):
    """
    Sort and group the input list of token sequences into batches, where each batch's
        total duration does not exceed the maximum.

    Args:
        tokens_list (List[List[int]]): A list of token sequences, where each inner
            list represents a sequence of tokens.
        max_duration (float): The maximum allowed total duration for each batch.
        prompt_duration (float): The duration cost per prompt in the batch.
        token_duration (float): The duration cost per token.

    Returns:
        batches: List[List[List[int]]]: A list of batches, where each batch is a list of
            token sequences that fit within the max duration.
        index: List[int]: The original index of each sentence, used to recover the
            sequential order in the future.
    """
    # Create index for each sentence
    indexed_tokens = list(enumerate(tokens_list))

    # Sort according to sentence length (for less padding)
    indexed_sorted_tokens = sorted(indexed_tokens, key=lambda x: len(x[1]))
    index = [indexed_sorted_tokens[i][0] for i in range(len(indexed_sorted_tokens))]
    sorted_tokens = [
        indexed_sorted_tokens[i][1] for i in range(len(indexed_sorted_tokens))
    ]

    batches = []
    batch = []
    batch_size = 0  # Total number of tokens in current batch

    for tokens in sorted_tokens:
        # Calculate if adding current token sequence would exceed max duration
        # Formula considers: existing tokens' duration + existing
        # prompts' duration + new tokens' duration
        if (
            batch_size * token_duration
            + len(batch) * prompt_duration
            + len(tokens) * token_duration
            <= max_duration
        ):
            # Add to current batch if within duration limit
            batch.append(tokens)
            batch_size += len(tokens)
        else:
            # If exceeding limit, finalize current batch (if not empty)
            if len(batch) > 0:
                batches.append(batch)
            # Start new batch with current token sequence
            batch = [tokens]
            batch_size = len(tokens)

    # Add the last batch if it's not empty
    if len(batch) > 0:
        batches.append(batch)

    return batches, index


def cross_fade_concat(
    chunks: List[torch.Tensor], fade_duration: float = 0.1, sample_rate: int = 24000
) -> torch.Tensor:
    """
    Concatenates audio chunks with cross-fading between consecutive chunks.

    Args:
        chunks: List of audio tensors, each with shape (C, T) where
                C = number of channel, T = time dimension (samples)
        fade_duration: Duration of cross-fade in seconds
        sample_rate: Audio sample rate in Hz

    Returns:
        Concatenated audio tensor with shape (N, T_total)
    """
    # Handle edge cases: empty input or single chunk
    if len(chunks) <= 1:
        return chunks[0] if chunks else torch.tensor([])

    # Calculate total fade samples from duration and sample rate
    fade_samples = int(fade_duration * sample_rate)

    # Use simple concatenation if fade duration is non-positive
    if fade_samples <= 0:
        return torch.cat(chunks, dim=-1)

    # Initialize final tensor with the first chunk
    final = chunks[0]

    # Iterate through remaining chunks to apply cross-fading
    for next_chunk in chunks[1:]:
        # Calculate safe fade length (cannot exceed either chunk's duration)
        k = min(fade_samples, final.shape[-1], next_chunk.shape[-1])

        # Fall back to simple concatenation if safe fade length is invalid
        if k <= 0:
            final = torch.cat([final, next_chunk], dim=-1)
            continue

        # Create fade curve (1 -> 0) with shape (1, k) for broadcasting
        fade = torch.linspace(1, 0, k, device=final.device)[None]

        # Concatenate three parts:
        # 1. Non-overlapping part of previous audio
        # 2. Cross-faded overlapping region
        # 3. Non-overlapping part of next audio
        final = torch.cat(
            [
                final[..., :-k],  # All samples except last k from previous
                final[..., -k:] * fade
                + next_chunk[..., :k] * (1 - fade),  # Cross-fade region
                next_chunk[..., k:],  # All samples except first k from next
            ],
            dim=-1,
        )

    return final


def add_punctuation(text: str):
    """Add punctuation if there is not in the end of text"""
    text = text.strip()
    if text[-1] not in punctuation:
        text += "."
    return text


def load_prompt_wav(prompt_wav: str, sampling_rate: int):
    """
    Load the waveform with torchaudio and resampling if needed.

    Parameters:
        prompt_wav: path of the prompt wav.
        sampling_rate: target sampling rate.

    Returns:
        Loaded prompt waveform with target sampling rate,
        PyTorch tensor of shape (C, T)
    """
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav)

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sampling_rate, new_freq=sampling_rate
        )
        prompt_wav = resampler(prompt_wav)
    return prompt_wav


def rms_norm(prompt_wav: torch.Tensor, target_rms: float):
    """
    Normalize the rms of prompt_wav is it is smaller than target rms.

    Parameters:
        prompt_wav: PyTorch tensor with shape (C, T).
        target_rms: target rms value

    Returns:
        prompt_wav: normalized prompt wav with shape (C, T).
        promt_rms: rms of original prompt wav. Will be used to
            re-normalize the generated wav.
    """
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms
    return prompt_wav, prompt_rms


def remove_silence(
    audio: torch.Tensor,
    sampling_rate: int,
    only_edge: bool = False,
    trail_sil: float = 0,
):
    """
    Remove silences longer than 1 second, and edge silences longer than 0.1 seconds

    Parameters:
        audio: PyTorch tensor with shape (C, T).
        sampling_rate: sampling rate of the audio.
        only_edge: If true, only remove edge silences.
        trail_sil: the duration of added trailing silence in ms.

    Returns:
        PyTorch tensor with shape (C, T), where C is number of channels
            and T is number of audio samples
    """
    # Load audio file
    wave = tensor_to_audiosegment(audio, sampling_rate)

    if not only_edge:
        # Split audio using silences longer than 1 second
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=1000,  # Silences longer than 1 second (1000ms)
            silence_thresh=-50,
            keep_silence=1000,  # Keep 1.0 second of silence around segments
            seek_step=10,
        )

        # Concatenate all non-silent segments
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    # Remove silence longer than 0.1 seconds in the begining and ending of wave
    wave = remove_silence_edges(wave, 100, -50)

    # Add trailing silence to avoid leaking prompt to generated speech.
    wave = wave + AudioSegment.silent(duration=trail_sil)

    # Convert to PyTorch tensor
    return audiosegment_to_tensor(wave)


def remove_silence_edges(
    audio: AudioSegment, keep_silence: int = 100, silence_threshold: float = -50
):
    """
    Remove edge silences longer than `keep_silence` ms.

    Parameters:
        audio: an AudioSegment object.
        keep_silence: kept silence in the edge.
        only_edge: If true, only remove edge silences.
        silence_threshold: the threshold of silence.

    Returns:
        An AudioSegment object
    """
    # Remove leading silence
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - keep_silence)
    audio = audio[start_idx:]

    # Remove trailing silence
    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - keep_silence)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def audiosegment_to_tensor(aseg):
    """
    Convert a pydub.AudioSegment to PyTorch audio tensor
    """
    audio_data = np.array(aseg.get_array_of_samples())

    # Convert to float32 and normalize to [-1, 1] range
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Handle channels
    if aseg.channels == 1:
        # Mono channel: add channel dimension (T) -> (1, T)
        tensor_data = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # Multi-channel: reshape to (C, T)
        tensor_data = torch.from_numpy(audio_data.reshape(-1, aseg.channels).T)

    return tensor_data


def tensor_to_audiosegment(tensor, sample_rate):
    """
    Convert a PyTorch audio tensor to pydub.AudioSegment

    Parameters:
        tensor: Tensor with shape (C, T), where C is the number of channels
            and T is the time steps
        sample_rate: Audio sample rate
    """
    # Convert tensor to numpy array
    audio_np = tensor.cpu().numpy()

    # Add channel dimension if single channel
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]

    # Convert to int16 type (common format for pydub)
    # Assumes tensor values are in [-1, 1] range as floating point
    audio_np = (audio_np * 32768.0).clip(-32768, 32767).astype(np.int16)

    # Convert to byte stream
    # For multi-channel audio, pydub requires interleaved format
    # (e.g., left-right-left-right)
    if audio_np.shape[0] > 1:
        # Convert to interleaved format
        audio_np = audio_np.transpose(1, 0).flatten()
    audio_bytes = audio_np.tobytes()

    # Create AudioSegment
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=tensor.shape[0],
    )

    return audio_segment
