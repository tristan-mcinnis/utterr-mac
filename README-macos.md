# Utterr macOS port

This is a practical macOS version of Utterr's real-time diarization timeline.

## What changed

- Replaced Windows-oriented `soundcard` loopback capture with `sounddevice` / CoreAudio input capture.
- Device picker now lists macOS input devices only.
- Added automatic resampling to 16 kHz before VAD / embedding.
- Kept the same core pipeline: Silero VAD → WeSpeaker embeddings via pyannote.audio → cosine clustering → PyQt timeline.
- CPU is the default model device for stability on macOS. You can try Apple Silicon MPS with `UTTERR_DEVICE=mps`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-macos.txt
python utterr_macos.py
```

The first launch downloads Silero and pyannote/WeSpeaker models, so it needs internet access.

## Microphone input

1. Run the app.
2. macOS will ask for microphone permission for Terminal / Python.
3. Select your microphone in **macOS Audio Input**.
4. Press **Start**.

If no input appears, grant permission in:

`System Settings → Privacy & Security → Microphone`

## System audio input

macOS does not expose Windows-style speaker loopback to ordinary apps. Use a virtual input driver:

```bash
brew install blackhole-2ch
```

Then:

1. Open **Audio MIDI Setup**.
2. Create a **Multi-Output Device** with your speakers/headphones + BlackHole 2ch.
3. Set macOS system output to that Multi-Output Device.
4. In Utterr, choose **BlackHole 2ch** as the input.
5. Press **Start**.

## Notes

- For live meetings, choose the meeting app's output as the Multi-Output Device, then capture BlackHole.
- For best diarization, use headphones or a clean virtual system-audio route; open speaker playback bleeding into a mic will degrade separation.
- `UTTERR_DEVICE=mps python utterr_macos.py` may work on Apple Silicon, but CPU mode is safer with pyannote dependencies.
