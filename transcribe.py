# 1. Accept an audio file
import argparse
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from faster_whisper import WhisperModel

RATE = 16_000


def read_args():
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--language", help="For example: en or ur")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.audio.resolve() == args.output.resolve():
        parser.error("Choose an output path different from the audio file.")
    return args


# 2. Transcribe spoken language into text
def recognize(model, samples, language):
    segments, _ = model.transcribe(
        samples, language=language, vad_filter=True
    )
    for segment in segments:
        text = segment.text.strip()
        if text:
            yield segment.start, segment.end, text


# 3. Return timestamps for each segment
def timed_record(number, start, end, text, offset_samples):
    offset_seconds = offset_samples / RATE
    return {
        "segment": number,
        "start_seconds": round(offset_seconds + start, 3),
        "end_seconds": round(offset_seconds + end, 3),
        "text": text,
    }


# 4. Handle different audio formats
@contextmanager
def decode_audio(path):
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-xerror",
        "-i", str(path.resolve()), "-map", "0:a:0",
        "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        with process.stdout as pcm:
            yield pcm
        if process.wait():
            raise ValueError(f"FFmpeg could not decode {path.name}")


# 5. Process long files and write the transcript
def audio_chunks(pcm):
    limit = 120 * RATE
    carry = b""
    while True:
        data = carry + pcm.read(2 * limit - len(carry))
        if not data:
            return
        samples = np.frombuffer(data, dtype="<i2").astype("float32") / 32768
        if len(samples) < limit:
            yield samples
            return

        # Cut midway through the quietest 100 ms in the final two seconds.
        width = RATE // 10
        tail = samples[-2 * RATE:].reshape(-1, width)
        quietest = int(np.argmin(np.mean(tail * tail, axis=1)))
        cut = limit - 2 * RATE + quietest * width + width // 2
        yield samples[:cut]
        carry = data[2 * cut:]


def run(args):
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    offset = 0
    number = 0
    with TemporaryDirectory(dir=args.output.parent) as directory:
        draft = Path(directory) / "transcript.jsonl"
        with draft.open("w", encoding="utf-8") as out, decode_audio(args.audio) as pcm:
            for samples in audio_chunks(pcm):
                for start, end, text in recognize(model, samples, args.language):
                    start = max(0, start)
                    end = min(end, len(samples) / RATE)
                    if end > start:
                        number += 1
                        row = timed_record(number, start, end, text, offset)
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                offset += len(samples)
        draft.replace(args.output)


if __name__ == "__main__":
    try:
        run(read_args())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Transcription failed: {exc}")
