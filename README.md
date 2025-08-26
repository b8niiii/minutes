# Meeting Minutes Pipeline

Utilities for converting meeting recordings into structured minutes.

## Setup

1. Install ffmpeg on your system.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python pipeline.py \
  --video meeting.mp4 \
  --chunk-seconds 180 \
  --max-speakers 6 \
  --attendees "Alessandro,Marta,Vasco,Marco"
```

Outputs are written to `out/`:

- `meeting.wav` (if audio extracted)
- `transcript_diarized.json`
- `chunks/chunk_XXX.json`
- `minutes_merged.json`
- `minutes.md`
- `minutes.docx`
- `actions.csv`

## Testing

Run the unit tests with:

```bash
pytest
```
