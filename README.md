# Audio transcription pipeline

This Python command-line tool takes a local recording and produces a UTF-8 JSON Lines transcript. Each line contains one recognized segment with start and end times in seconds from the beginning of the decoded audio.

## Run the example

Install FFmpeg on your PATH and use a Python 3.10 or 3.11 environment. On macOS, Homebrew users can run `brew install ffmpeg`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python transcribe.py examples/meeting.wav -o transcript.jsonl --language en
cat transcript.jsonl
```

The first run downloads the multilingual Whisper base model. A pre-downloaded faster-whisper model directory can be supplied with `--model`. Replace the example path with a WAV or MP3 recording. Use `--language ur` for Urdu or omit the option to detect the language per chunk.

## Implementation decisions

| Decision | Reason and trade-off |
| --- | --- |
| Python CLI with one CPU-loaded model | A reviewer can run the example locally without a GPU or a hosted speech API. CPU INT8 inference keeps setup small. Processing time depends on the recording length and the machine. |
| Stream decoded samples from FFmpeg | The first audio track of each input is converted to 16 kHz mono PCM. The recognizer receives the same type of input for WAV and MP3. Downmixing loses channel separation. |
| Read up to 120 seconds of audio at a time | Memory use for decoded audio stays bounded. The code cuts within the quietest 100 ms interval in the final two seconds and carries the remaining samples forward. A word can still cross the boundary during continuous speech. |
| Track the number of processed samples | Adding the sample offset divided by 16,000 converts chunk-local times to positions in the source recording, including silence. Segment boundaries are model estimates. |
| Write JSON Lines into a temporary file | Downstream consumers can read a line at a time. The script replaces the destination only after decoding and recognition finish. If a run fails, an existing transcript is preserved. |

## Check the output

The included WAV contains synthesized speech saying “The meeting starts at ten. Please bring the project report.” To check that timestamps carry across a chunk boundary, add 125 seconds of silence before the example.

```bash
ffmpeg -i examples/meeting.wav -af 'adelay=125000:all=1' delayed.mp3
python transcribe.py delayed.mp3 -o delayed.jsonl --language en
cat delayed.jsonl
```

Look for the phrase near 125 seconds. A previous run returned the following segment. Exact text and timing may vary by model and version.

```json
{"segment":1,"start_seconds":124.85,"end_seconds":128.85,"text":"The meeting starts at 10. Please bring the project report."}
```

## Proposed multi-user service

The delivered implementation is the CLI. The following design would extend it into a hosted service.

**Concurrent uploads.** An authenticated API would create a job with a unique ID and return a short-lived upload URL for a private object. Once the upload is complete, the API would verify the object and mark the job queued. Workers would claim queued jobs from PostgreSQL using short transactions and row locks with SKIP LOCKED. They would release the transaction before running recognition. A worker concurrency limit would keep simultaneous inferences within CPU and memory capacity.

**Storage.** Private object storage would hold original audio and JSONL transcripts. A PostgreSQL job row would record the owner, input object version, output key, status, model version, language, and attempt count. The API would verify ownership before issuing a download link. A retention policy could expire source audio separately from transcripts.

**Recovery.** A worker would renew a time-limited lease while processing. If a worker stopped, another could reclaim the job after the lease expired. Temporary failures would be retried up to three times with exponential backoff and jitter. Invalid audio would fail immediately with a clear reason. Each attempt would write to a separate output key. Only the current lease holder could publish that key and mark the job complete, preventing a late retry from overwriting a finished result. The first service version would retry the whole recording.

**HTTP API.** POST /transcriptions would create the upload and return a job ID and upload URL (201). POST /transcriptions/{id}/complete would verify the uploaded object and queue the job (202). Repeated completion requests would return the existing state. GET /transcriptions/{id} would return the job state, a result link on success, or a reason on failure. An owner-scoped idempotency key on creation would let clients safely retry after a network timeout.
