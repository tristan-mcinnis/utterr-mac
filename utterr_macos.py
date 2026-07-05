"""Utterr macOS port: PyQt real-time speaker timeline using sounddevice/CoreAudio.

Microphone input works directly. System audio on macOS must be routed through a
virtual input device such as BlackHole or Loopback, then selected in the app.
"""
from __future__ import annotations

import math, os, queue, sys, time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
PROCESS_EVERY_SEC = 0.15
MAX_SPEAKERS = 10
SIM_THRESHOLD = 0.35
UPDATE_THRESHOLD = 0.45
PENDING_COLOR = "#888888"
SPEAKER_COLORS = ["#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF", "#44FFFF", "#FF8844", "#FF009D", "#8844FF", "#FFAA44"]
DEVICE_PREF = os.environ.get("UTTERR_DEVICE", "cpu").lower()


def torch_device() -> str:
    if DEVICE_PREF == "cuda" and torch.cuda.is_available():
        return "cuda"
    if DEVICE_PREF == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v if n == 0 or not np.isfinite(n) else v / n


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == SAMPLE_RATE or audio.size == 0:
        return audio
    g = math.gcd(int(sr), SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)


@dataclass
class Segment:
    start: float
    duration: float
    speaker: Optional[int | str] = None
    speech: bool = False

    @property
    def end(self) -> float:
        return self.start + self.duration


class SileroVAD(QThread):
    loaded = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.inq, self.outq = queue.Queue(), queue.Queue()
        self.model = None
        self.get_speech_ts = None
        self.ready = False
        self.stop_flag = False

    def run(self):
        self.status.emit("Loading Silero VAD...")
        self.model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False, trust_repo=True)
        self.model = self.model.to("cpu")
        self.get_speech_ts = utils[0]
        self.ready = True
        self.status.emit("Silero VAD loaded")
        self.loaded.emit()
        while not self.stop_flag:
            try:
                task_id, audio = self.inq.get(timeout=0.1)
                with torch.no_grad():
                    ts = self.get_speech_ts(torch.from_numpy(audio.astype(np.float32)), self.model, threshold=0.5, sampling_rate=SAMPLE_RATE, return_seconds=False)
                self.outq.put((task_id, len(ts) > 0))
            except queue.Empty:
                pass
            except Exception as e:
                self.status.emit(f"VAD error: {e}")

    def submit(self, audio: np.ndarray):
        if not self.ready:
            return None
        task_id = time.time_ns()
        self.inq.put((task_id, audio.copy()))
        return task_id

    def result(self):
        try:
            return self.outq.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stop_flag = True


class WeSpeaker(QThread):
    loaded = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.inq, self.outq = queue.Queue(), queue.Queue()
        self.inference = None
        self.ready = False
        self.stop_flag = False

    def run(self):
        self.status.emit(f"Loading WeSpeaker on {self.device.upper()}...")
        from pyannote.audio import Inference, Model
        model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=False).to(torch.device(self.device))
        self.inference = Inference(model, window="whole")
        self.ready = True
        self.status.emit("WeSpeaker loaded")
        self.loaded.emit()
        while not self.stop_flag:
            try:
                task_id, audio = self.inq.get(timeout=0.1)
                wave = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    emb = self.inference({"waveform": wave, "sample_rate": SAMPLE_RATE})
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                self.outq.put((task_id, norm(np.asarray(emb))))
            except queue.Empty:
                pass
            except Exception as e:
                self.status.emit(f"Embedding error: {e}")

    def submit(self, audio: np.ndarray):
        if not self.ready:
            return None
        task_id = time.time_ns()
        self.inq.put((task_id, audio.copy()))
        return task_id

    def result(self):
        try:
            return self.outq.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stop_flag = True


class SpeakerBook:
    def __init__(self):
        self.embs = [[] for _ in range(MAX_SPEAKERS)]
        self.means: list[Optional[np.ndarray]] = [None] * MAX_SPEAKERS
        self.active: set[int] = set()

    def reset(self):
        self.__init__()

    def classify(self, emb: np.ndarray):
        emb = norm(emb)
        if not self.active:
            self.active.add(0); self.embs[0].append(emb); self.means[0] = emb
            return 0, 1.0
        ids = sorted(self.active)
        sims = [float(np.dot(self.means[i], emb)) for i in ids if self.means[i] is not None]
        best_pos = int(np.argmax(sims)); best_id = ids[best_pos]; best_sim = sims[best_pos]
        if best_sim < SIM_THRESHOLD and len(self.active) < MAX_SPEAKERS:
            new_id = next(i for i in range(MAX_SPEAKERS) if i not in self.active)
            self.active.add(new_id); self.embs[new_id].append(emb); self.means[new_id] = emb
            return new_id, best_sim
        if best_sim >= UPDATE_THRESHOLD:
            self.embs[best_id].append(emb)
            self.means[best_id] = norm(np.median(np.array(self.embs[best_id]), axis=0))
        return best_id, best_sim


class AudioCapture(QThread):
    chunk = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, device_id=None):
        super().__init__()
        self.device_id = device_id
        self.paused = True
        self.running = True

    @staticmethod
    def devices():
        out = []
        try:
            for idx, d in enumerate(sd.query_devices()):
                ch = int(d.get("max_input_channels", 0))
                if ch > 0:
                    out.append((idx, d.get("name", f"Input {idx}"), ch, int(d.get("default_samplerate", 48000))))
        except Exception as e:
            print(e)
        return out

    @staticmethod
    def default_device():
        try:
            d = int(sd.default.device[0])
            return d if d >= 0 else None
        except Exception:
            return None

    def run(self):
        dev = self.device_id if self.device_id is not None else self.default_device()
        if dev is None:
            self.status.emit("No audio input device found")
            return
        info = sd.query_devices(dev, "input")
        sr = int(info["default_samplerate"])
        channels = max(1, min(int(info["max_input_channels"]), 2))
        self.status.emit(f"Audio input ready: {info['name']}")

        def callback(indata, frames, time_info, stat):
            if stat:
                self.status.emit(str(stat))
            if self.paused or not self.running:
                return
            audio = np.asarray(indata, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = to_16k(audio, sr)
            m = float(np.max(np.abs(audio))) if audio.size else 0.0
            if m > 1.0:
                audio = audio / m
            self.chunk.emit(audio.astype(np.float32))

        with sd.InputStream(device=dev, channels=channels, samplerate=sr, blocksize=max(256, int(sr * 0.1)), dtype="float32", callback=callback):
            while self.running:
                self.msleep(50)

    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def stop(self): self.running = False


class Timeline:
    def __init__(self):
        self.start_time = None
        self.paused_at = None
        self.paused_total = 0.0
        self.segments: list[Segment] = []

    def start(self):
        if self.start_time is None:
            self.start_time = time.time(); self.paused_at = None; self.paused_total = 0.0

    def pause(self):
        if self.start_time is not None and self.paused_at is None:
            self.paused_at = time.time()

    def resume(self):
        if self.paused_at is not None:
            self.paused_total += time.time() - self.paused_at; self.paused_at = None

    def now(self):
        if self.start_time is None:
            return 0.0
        t = self.paused_at if self.paused_at is not None else time.time()
        return t - self.start_time - self.paused_total

    def add(self, seg: Segment):
        if self.start_time is not None and self.paused_at is None:
            self.segments.append(seg)

    def reset(self): self.__init__()


class Processor:
    def __init__(self, vad: SileroVAD, enc: WeSpeaker, speakers: SpeakerBook, timeline: Timeline, widget):
        self.vad, self.enc, self.speakers, self.timeline, self.widget = vad, enc, speakers, timeline, widget
        self.paused = True
        self.buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self.idx = 0; self.full = False; self.last = 0.0
        self.vad_wait = {}; self.emb_wait = {}

    def add_chunk(self, audio: np.ndarray):
        if self.paused: return
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size >= WINDOW_SAMPLES:
            self.buf[:] = audio[-WINDOW_SAMPLES:]; self.idx = 0; self.full = True
        else:
            end = self.idx + audio.size
            if end <= WINDOW_SAMPLES:
                self.buf[self.idx:end] = audio; self.idx = end % WINDOW_SAMPLES; self.full |= end == WINDOW_SAMPLES
            else:
                first = WINDOW_SAMPLES - self.idx
                self.buf[self.idx:] = audio[:first]; self.buf[:audio.size-first] = audio[first:]
                self.idx = audio.size - first; self.full = True
        now = time.time()
        if now - self.last >= PROCESS_EVERY_SEC:
            self.last = now; self.process_window()
        self.drain()

    def window(self):
        if not self.full:
            return self.buf[:self.idx] if self.idx >= SAMPLE_RATE // 2 else None
        return np.concatenate([self.buf[self.idx:], self.buf[:self.idx]])

    def process_window(self):
        w = self.window()
        if w is None: return
        seg = Segment(max(0, self.timeline.now() - WINDOW_SEC), WINDOW_SEC)
        tid = self.vad.submit(w)
        if tid: self.vad_wait[tid] = (seg, w.copy())

    def drain(self):
        while True:
            r = self.vad.result()
            if r is None: break
            tid, is_speech = r
            if tid not in self.vad_wait: continue
            seg, audio = self.vad_wait.pop(tid); seg.speech = is_speech
            if is_speech:
                eid = self.enc.submit(audio)
                if eid: self.emb_wait[eid] = seg
            else:
                self.timeline.add(seg)
        while True:
            r = self.enc.result()
            if r is None: break
            tid, emb = r
            if tid not in self.emb_wait: continue
            seg = self.emb_wait.pop(tid)
            spk, _ = self.speakers.classify(emb)
            seg.speaker = spk
            self.timeline.add(seg)
        self.widget.update_segments(self.timeline.segments)

    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def reset(self):
        self.speakers.reset(); self.timeline.reset(); self.vad_wait.clear(); self.emb_wait.clear(); self.idx = 0; self.full = False; self.widget.update_segments([])


class TimelineWidget(QWidget):
    def __init__(self):
        super().__init__(); self.segments = []; self.px = 100; self.setMinimumHeight(560); self.setMinimumWidth(900)

    def update_segments(self, segments):
        self.segments = list(segments)
        max_t = max((s.end for s in self.segments), default=0)
        self.setMinimumWidth(max(900, int((max_t + 5) * self.px)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, self.width(), self.height(), QBrush(QColor("#1A1A1A")))
        layers = MAX_SPEAKERS + 1; h = max(36, self.height() // layers)
        p.setPen(QPen(QColor("#444")))
        for i in range(layers): p.drawLine(0, i*h, self.width(), i*h)
        p.setFont(QFont("Arial", 10)); p.setPen(QPen(QColor("#AAA")))
        for sec in range(0, int(max((s.end for s in self.segments), default=0)) + 10, 10):
            x = sec * self.px; p.drawLine(x, 0, x, self.height()); p.drawText(x + 4, 14, f"{sec//60:02d}:{sec%60:02d}")
        for s in self.segments:
            x1, x2 = int(s.start * self.px), int(s.end * self.px)
            if not s.speech:
                layer, color = MAX_SPEAKERS, QColor("#666")
            else:
                layer = int(s.speaker or 0) % MAX_SPEAKERS
                color = QColor(SPEAKER_COLORS[layer])
            color.setAlpha(130)
            p.fillRect(x1, layer*h + 5, max(1, x2-x1), h - 10, QBrush(color))
        p.setPen(QPen(QColor("#DDD"))); p.setFont(QFont("Arial", 12))
        for i in range(MAX_SPEAKERS): p.drawText(10, i*h + h//2 + 5, f"Speaker {i+1}")
        p.drawText(10, MAX_SPEAKERS*h + h//2 + 5, "Non-speech")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Utterr macOS")
        self.vad = SileroVAD(); self.enc = WeSpeaker(torch_device())
        self.capture = None; self.processor = None; self.ready = {"vad": False, "enc": False}; self.recording = False
        self.speakers = SpeakerBook(); self.timeline_state = Timeline()
        self.build_ui(); self.wire_models(); QTimer.singleShot(300, self.start_models)

    def build_ui(self):
        root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
        self.status = QLabel("Preparing..."); layout.addWidget(self.status)
        self.timeline = TimelineWidget(); self.scroll = QScrollArea(); self.scroll.setWidget(self.timeline); self.scroll.setWidgetResizable(True); self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn); layout.addWidget(self.scroll, 1)
        row = QHBoxLayout(); self.start_btn = QPushButton("Start"); self.start_btn.setEnabled(False); self.start_btn.clicked.connect(self.toggle); row.addWidget(self.start_btn)
        self.reset_btn = QPushButton("Reset"); self.reset_btn.setEnabled(False); self.reset_btn.clicked.connect(self.reset); row.addWidget(self.reset_btn)
        self.device_box = QComboBox(); self.refresh_devices(); row.addWidget(QLabel("Input:")); row.addWidget(self.device_box, 1)
        self.apply_btn = QPushButton("Apply"); self.apply_btn.setEnabled(False); self.apply_btn.clicked.connect(self.apply_device); row.addWidget(self.apply_btn)
        layout.addLayout(row)
        self.setStyleSheet("QWidget{background:#2D2D30;color:#DDD} QPushButton,QComboBox{background:#3F3F46;color:#EEE;border:1px solid #666;padding:6px} QLabel{padding:4px}")
        self.resize(1300, 850)

    def wire_models(self):
        self.vad.status.connect(self.set_status); self.enc.status.connect(self.set_status)
        self.vad.loaded.connect(lambda: self.model_ready("vad")); self.enc.loaded.connect(lambda: self.model_ready("enc"))

    def start_models(self): self.vad.start(); self.enc.start()
    def set_status(self, s): self.status.setText(s); print(s)

    def refresh_devices(self):
        self.device_box.clear(); default = AudioCapture.default_device()
        for idx, name, ch, sr in AudioCapture.devices():
            self.device_box.addItem(f"[{idx}] {name} — {ch}ch @ {sr} Hz", idx)
            if idx == default: self.device_box.setCurrentIndex(self.device_box.count() - 1)

    def model_ready(self, which):
        self.ready[which] = True
        if all(self.ready.values()):
            self.processor = Processor(self.vad, self.enc, self.speakers, self.timeline_state, self.timeline)
            self.create_capture(); self.start_btn.setEnabled(True); self.reset_btn.setEnabled(True); self.apply_btn.setEnabled(True); self.set_status("Ready — choose mic or BlackHole input, then Start")

    def create_capture(self):
        if self.capture: self.capture.stop(); self.capture.wait(1000)
        self.capture = AudioCapture(self.device_box.currentData()); self.capture.status.connect(self.set_status); self.capture.chunk.connect(self.processor.add_chunk); self.capture.start()

    def apply_device(self):
        was = self.recording
        if was: self.toggle()
        self.create_capture()
        if was: QTimer.singleShot(200, self.toggle)

    def toggle(self):
        if not self.recording:
            self.timeline_state.start(); self.timeline_state.resume(); self.capture.resume(); self.processor.resume(); self.recording = True; self.start_btn.setText("Pause"); self.set_status("Recording...")
        else:
            self.timeline_state.pause(); self.capture.pause(); self.processor.pause(); self.recording = False; self.start_btn.setText("Resume"); self.set_status("Paused")

    def reset(self):
        if self.processor: self.processor.reset()
        self.recording = False; self.start_btn.setText("Start"); self.set_status("Reset")

    def closeEvent(self, event):
        if self.capture: self.capture.stop(); self.capture.wait(1000)
        self.vad.stop(); self.enc.stop(); self.vad.wait(1000); self.enc.wait(1000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())


if __name__ == "__main__": main()
