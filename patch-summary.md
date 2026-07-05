# Patch summary

Main repo reference: `maximus-choi/Utterr`, recommended file `rt_timeline_pyannote.py`.

The original app imports `soundcard as sc` and uses `sc.get_microphone(..., include_loopback=True)` to capture Windows speaker loopback. That is the Windows-specific part.

The macOS port replaces that layer with:

- `import sounddevice as sd`
- `sd.query_devices()` for input-device enumeration
- `sd.InputStream(...)` for CoreAudio capture
- `scipy.signal.resample_poly(...)` to convert native 44.1/48 kHz macOS device audio to the 16 kHz expected by Silero/WeSpeaker

System audio is not captured from output devices directly. On macOS it must appear as an input device via BlackHole, Loopback, or another virtual audio driver.
