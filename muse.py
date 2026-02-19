# ===========================================
# Muse Focus LAB PRO (Single-file)
# - Live EEG + (optional) PPG/HRV
# - ML Focus Model (personalized calibration)
# - True Biofeedback (game speed + UI color from EEG/ML)
# - Export: CSV + Excel + PDF report
# - Dashboard inside GUI (live charts + post-session summary)
# - Pause/Resume (safe, low CPU)
#
# Requirements:
# pip install bleak numpy pandas scipy openpyxl pyttsx3 matplotlib reportlab
# ===========================================

import asyncio
import threading
import queue
import logging
from logging.handlers import RotatingFileHandler
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import pyttsx3
from bleak import BleakClient, BleakScanner
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas


# =========================================================
# Logging
# =========================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("MuseFocusLabPro")
    logger.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "muse_focus_lab_pro.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info("Logging started.")
    return logger


LOGGER = setup_logging()


# =========================================================
# Muse 2 UUIDs (EEG) + Optional PPG
# =========================================================
UUID_CONTROL = "273e0001-4c4d-454d-96be-f03bac821358"
UUID_TP9  = "273e0003-4c4d-454d-96be-f03bac821358"
UUID_AF7  = "273e0004-4c4d-454d-96be-f03bac821358"
UUID_AF8  = "273e0005-4c4d-454d-96be-f03bac821358"
UUID_TP10 = "273e0006-4c4d-454d-96be-f03bac821358"

EEG_UUIDS = [UUID_TP9, UUID_AF7, UUID_AF8, UUID_TP10]
EEG_CH    = ["TP9", "AF7", "AF8", "TP10"]

# Muse PPG UUID can vary across firmware; this is commonly used in Muse BLE profiles.
# If it fails, we continue without HRV (no crash).
UUID_PPG = "273e000f-4c4d-454d-96be-f03bac821358"


# =========================================================
# Config
# =========================================================
FS_EEG = 256
FS_PPG = 64  # approximate; if PPG stream exists

WARMUP_SEC = 2
NOTCH_HZ = 50.0
EEG_BANDPASS = (1.0, 45.0)

# Leave empty to auto-find by name
MUSE_ADDRESS = ""

# Protocol (your request)
PHASES = [
    ("OE1",   "افتح عينيك", 30, "#1abc9c"),
    ("REST1", "استرح",      10, "#95a5a6"),
    ("CE",    "اغلق عينيك", 30, "#3498db"),
    ("REST2", "استرح",      10, "#95a5a6"),
    ("OE2",   "افتح عينيك", 30, "#1abc9c"),
]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "smr":   (12, 15),
    "beta":  (15, 30),
    "gamma": (30, 45),
}

# Simple quality thresholds
MAX_PTP_UV = 250.0
MAX_RMS_UV = 60.0
FLATLINE_STD_UV = 0.2

# Live window for features and focus (seconds)
LIVE_FEATURE_WIN_SEC = 2

# Export summary windows
FEATURE_BLOCK_SEC = 4


# =========================================================
# Helpers: TTS
# =========================================================
def init_tts() -> pyttsx3.Engine:
    eng = pyttsx3.init()
    eng.setProperty("rate", 150)
    return eng


# =========================================================
# EEG decode (Muse 20 bytes => 12 samples)
# =========================================================
def unpack_12bit_samples(payload18: bytes) -> List[int]:
    out = []
    for i in range(0, 18, 3):
        a = payload18[i]
        b = payload18[i + 1]
        c = payload18[i + 2]
        s1 = (a << 4) | (b >> 4)
        s2 = ((b & 0x0F) << 8) | c
        out.extend([s1, s2])
    return out

def eeg_to_uv(v12: int) -> float:
    return (v12 - 2048) * 0.48828125

def decode_eeg_packet(raw20: bytes) -> List[float]:
    payload18 = raw20[2:20]
    vals12 = unpack_12bit_samples(payload18)
    return [eeg_to_uv(v) for v in vals12]


# =========================================================
# Optional PPG decode (best-effort)
# Note: PPG payload formats can vary; we keep it robust and permissive.
# We'll treat data bytes as signed-ish integers and build a stream.
# =========================================================
def decode_ppg_packet(data: bytes) -> List[float]:
    # Many devices send 3-channel PPG; we simplify to a single stream:
    # take bytes and convert to centered values.
    if not data:
        return []
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    arr = arr - np.mean(arr)
    # downsample by grouping if too long
    if len(arr) > 32:
        arr = arr.reshape(-1, 2).mean(axis=1)
    return arr.tolist()


# =========================================================
# Signal processing
# =========================================================
def butter_bandpass_sos(low: float, high: float, fs: int, order: int = 4):
    return butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

def preprocess_eeg(x: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    if not np.isfinite(med):
        med = 0.0
    x = np.nan_to_num(x, nan=med, posinf=med, neginf=med)

    # Remove DC
    x = x - np.median(x)

    # Bandpass
    sos = butter_bandpass_sos(EEG_BANDPASS[0], EEG_BANDPASS[1], fs, order=4)
    x = sosfiltfilt(sos, x)

    # Notch
    b, a = iirnotch(w0=NOTCH_HZ, Q=30, fs=fs)
    x = filtfilt(b, a, x)
    return x

def bandpower_psd(x: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), fs * 2))
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapezoid(pxx[idx], f[idx]))

def compute_band_features(x: np.ndarray, fs: int) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    tp = bandpower_psd(x, fs, 1, 45)
    feats["total_1_45"] = tp

    for band, (f1, f2) in BANDS.items():
        bp = bandpower_psd(x, fs, f1, f2)
        feats[f"abs_{band}"] = bp
        feats[f"rel_{band}"] = bp / (tp + 1e-12)

    feats["rms_uv"] = float(np.sqrt(np.mean(x**2)))
    feats["ptp_uv"] = float(np.ptp(x))
    feats["std_uv"] = float(np.std(x))
    feats["artifact_flag"] = float((feats["ptp_uv"] > MAX_PTP_UV) or (feats["rms_uv"] > MAX_RMS_UV))
    feats["flatline_flag"] = float(feats["std_uv"] < FLATLINE_STD_UV)
    return feats

def lr_avg(per_ch: Dict[str, Dict[str, float]], key: str) -> Tuple[float, float]:
    L = (per_ch["TP9"][key] + per_ch["AF7"][key]) / 2.0
    R = (per_ch["AF8"][key] + per_ch["TP10"][key]) / 2.0
    return float(L), float(R)

def safe_ratio(a: float, b: float) -> float:
    return float(a / (b + 1e-12))


# =========================================================
# HRV (simple time-domain)
# =========================================================
def _simple_peak_detect(x: np.ndarray, min_dist: int = 10) -> List[int]:
    # Very simple local maxima detector with minimum distance
    peaks: List[int] = []
    last = -10**9
    for i in range(1, len(x) - 1):
        if i - last < min_dist:
            continue
        if x[i] > x[i-1] and x[i] > x[i+1]:
            peaks.append(i)
            last = i
    return peaks

def compute_hrv_from_ppg(ppg: np.ndarray, fs: int) -> Dict[str, float]:
    # Best-effort. If not enough stable peaks => empty.
    if ppg.size < fs * 8:
        return {}
    x = ppg.astype(float)
    x = x - np.mean(x)
    # light smoothing (moving average)
    if len(x) > 5:
        x = np.convolve(x, np.ones(5)/5, mode="same")

    peaks = _simple_peak_detect(x, min_dist=int(0.35 * fs))  # ~>=170 bpm max
    if len(peaks) < 5:
        return {}

    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if rr.size < 4:
        return {}

    hr = 60.0 / float(np.mean(rr))
    sdnn = float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2))) if rr.size >= 3 else float("nan")

    return {"HR": hr, "SDNN": sdnn, "RMSSD": rmssd}


# =========================================================
# ML model (Ridge Regression, implemented with numpy)
# - Train on calibration points: features -> target (self rating 1..10)
# - Predict continuous focus 0..100
# =========================================================
class RidgeModel:
    def __init__(self, n_features: int, alpha: float = 10.0):
        self.n = n_features
        self.alpha = alpha
        self.w = np.zeros((n_features,), dtype=float)
        self.b = 0.0
        self._trained = False

    @property
    def trained(self) -> bool:
        return self._trained

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # X: (m, n), y: (m,)
        # Ridge with bias: solve for w,b by augmenting column of 1s
        m, n = X.shape
        assert n == self.n

        Xc = X.copy()
        yc = y.copy()

        # Standardize features for stability
        self.mu = Xc.mean(axis=0)
        self.sigma = Xc.std(axis=0) + 1e-9
        Xs = (Xc - self.mu) / self.sigma

        # augment bias
        Xa = np.hstack([Xs, np.ones((m, 1))])
        I = np.eye(n + 1)
        I[-1, -1] = 0.0  # do not regularize bias
        A = Xa.T @ Xa + self.alpha * I
        b = Xa.T @ yc
        sol = np.linalg.solve(A, b)

        self.w = sol[:n]
        self.b = sol[-1]
        self._trained = True

    def predict(self, x: np.ndarray) -> float:
        # x: (n,)
        if not self._trained:
            return float("nan")
        xs = (x - self.mu) / self.sigma
        return float(xs @ self.w + self.b)


# =========================================================
# BLE discovery
# =========================================================
async def find_muse_address(retries: int = 6, timeout: float = 6.0) -> Optional[str]:
    for attempt in range(1, retries + 1):
        devices = await BleakScanner.discover(timeout=timeout)

        if MUSE_ADDRESS:
            for d in devices:
                if d.address.lower() == MUSE_ADDRESS.lower():
                    LOGGER.info("Muse found by address: %s (%s)", d.name, d.address)
                    return d.address

        for d in devices:
            if "Muse" in (d.name or ""):
                LOGGER.info("Muse found by name: %s (%s)", d.name, d.address)
                return d.address

        LOGGER.warning("Muse not found (attempt %d/%d).", attempt, retries)
        await asyncio.sleep(1.5)
    return None


# =========================================================
# Games (Biofeedback)
# =========================================================
class FocusGameBase:
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.running = False
        self.speed_factor = 1.0  # updated by biofeedback (focus)
        self.w = int(canvas["width"])
        self.h = int(canvas["height"])

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        self.canvas.delete("all")

    def set_speed(self, f: float) -> None:
        self.speed_factor = float(np.clip(f, 0.5, 3.0))

    def tick(self) -> None:
        raise NotImplementedError


class GameMovingDot(FocusGameBase):
    def __init__(self, canvas: tk.Canvas):
        super().__init__(canvas)
        self.r = 14
        self.pos = [self.w // 2, self.h // 2]
        self.vel = [5, 4]

    def start(self) -> None:
        super().start()
        self.pos = [self.w // 2, self.h // 2]
        self.vel = [random.choice([-6, 6]), random.choice([-5, 5])]

    def tick(self) -> None:
        if not self.running:
            return
        self.canvas.delete("dot")
        x, y = self.pos
        r = self.r
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#e74c3c", outline="", tags="dot")

        vx = self.vel[0] * self.speed_factor
        vy = self.vel[1] * self.speed_factor
        self.pos[0] += vx
        self.pos[1] += vy

        if self.pos[0] < r or self.pos[0] > self.w - r:
            self.vel[0] *= -1
        if self.pos[1] < r or self.pos[1] > self.h - r:
            self.vel[1] *= -1


class GameSaccadeTargets(FocusGameBase):
    def __init__(self, canvas: tk.Canvas):
        super().__init__(canvas)
        self.r = 16
        self.counter = 0
        self.hold_frames = 20

    def tick(self) -> None:
        if not self.running:
            return
        self.counter += 1
        # higher focus -> change faster (shorter hold)
        hold = int(max(6, self.hold_frames / self.speed_factor))
        if self.counter % hold != 0:
            return

        self.canvas.delete("tgt")
        corners = [
            (self.r, self.r),
            (self.w - self.r, self.r),
            (self.r, self.h - self.r),
            (self.w - self.r, self.h - self.r),
            (self.w // 2, self.h // 2),
        ]
        x, y = random.choice(corners)
        r = self.r
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#f1c40f", outline="", tags="tgt")


class GameBreathingCircle(FocusGameBase):
    def __init__(self, canvas: tk.Canvas):
        super().__init__(canvas)
        self.t = 0.0

    def tick(self) -> None:
        if not self.running:
            return
        # in breathing game, speed_factor slows breathing if focus is low (calming)
        self.t += 0.06 / self.speed_factor
        self.canvas.delete("b")
        cx, cy = self.w // 2, self.h // 2
        r = 25 + 60 * (0.5 * (1 + math.sin(self.t)))
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="#2ecc71", width=4, tags="b")


# Requested: Game 4 (circular track + adaptive speed)
class GameCircularTrack(FocusGameBase):
    def __init__(self, canvas: tk.Canvas):
        super().__init__(canvas)
        self.t = 0.0
        self.r_dot = 10
        self.track_r = min(self.w, self.h) * 0.35

    def tick(self) -> None:
        if not self.running:
            return
        self.canvas.delete("c")
        cx, cy = self.w // 2, self.h // 2

        # draw track
        tr = self.track_r
        self.canvas.create_oval(cx-tr, cy-tr, cx+tr, cy+tr, outline="#34495e", width=2, tags="c")

        # speed follows focus
        self.t += 0.08 * self.speed_factor
        x = cx + tr * math.cos(self.t)
        y = cy + tr * math.sin(self.t)

        r = self.r_dot
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#9b59b6", outline="", tags="c")


# =========================================================
# Responsiveness score (simple, interpretable)
# - Alpha suppression: alpha_CE - alpha_OE  (should rise in CE)
# - Beta reactivity: beta_OE - beta_CE (varies; task dependent)
# - Stability: low artifact rate
# Returns 0..100
# =========================================================
def compute_responsiveness(phase_summary: pd.DataFrame) -> float:
    # requires rows for OE1, CE, OE2 in phase_summary
    try:
        oe = phase_summary[phase_summary["phase"].isin(["OE1", "OE2"])].mean(numeric_only=True)
        ce = phase_summary[phase_summary["phase"] == "CE"].iloc[0]

        alpha_sup = float(ce.get("rel_alpha_avg", np.nan) - oe.get("rel_alpha_avg", np.nan))
        beta_reac = float(oe.get("rel_beta_avg", np.nan) - ce.get("rel_beta_avg", np.nan))
        art = float(phase_summary["artifact_rate_avg"].mean())

        # map
        s1 = np.clip((alpha_sup + 0.02) / 0.08, 0, 1)  # heuristic scaling
        s2 = np.clip((beta_reac + 0.01) / 0.06, 0, 1)
        s3 = 1.0 - np.clip(art, 0, 1)

        score = 100.0 * (0.45*s1 + 0.35*s2 + 0.20*s3)
        return float(np.clip(score, 0, 100))
    except Exception:
        return float("nan")


# =========================================================
# GUI + BLE + Dashboard + Export
# =========================================================
@dataclass
class LiveMetrics:
    focus_raw: float = float("nan")
    focus_ml: float = float("nan")
    quality: str = "collecting…"
    artifact_rate: float = float("nan")
    flat_rate: float = float("nan")
    hr: float = float("nan")
    rmssd: float = float("nan")
    sdnn: float = float("nan")


class MuseFocusLabProApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Muse Focus Lab PRO (ML + Biofeedback)")
        self.root.geometry("1250x780")
        self.root.configure(bg="#0f1115")

        self.tts = init_tts()

        # run controls
        self._running = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # running by default

        # live buffers
        self.plot_window_sec = 10
        self.max_plot_samples = FS_EEG * self.plot_window_sec
        from collections import deque
        self.live_eeg = {ch: deque(maxlen=self.max_plot_samples) for ch in EEG_CH}
        self.live_ppg = deque(maxlen=FS_PPG * 30)
        self.live_ts = deque(maxlen=self.max_plot_samples)

        # packet buffers for export
        self.raw_packets_eeg: List[Dict[str, Any]] = []
        self.raw_samples_eeg: List[Dict[str, Any]] = []
        self.raw_samples_ppg: List[Dict[str, Any]] = []

        # phase tracking for export
        self.phase_timeline: List[Dict[str, Any]] = []  # phase start/end timestamps
        self.current_phase: str = "READY"

        # ML calibration data
        self.feature_names = self._build_feature_schema()
        self.model = RidgeModel(n_features=len(self.feature_names), alpha=15.0)
        self.calib_X: List[np.ndarray] = []
        self.calib_y: List[float] = []

        # live derived metrics
        self.metrics = LiveMetrics()
        self.focus_history: List[Dict[str, Any]] = []

        # queue from BLE thread -> GUI
        self.q: "queue.Queue[Tuple[str, datetime, Any]]" = queue.Queue()

        # game
        self.game: Optional[FocusGameBase] = None

        # dashboard figures
        self.fig = None
        self.ax = None
        self.lines = {}

        self.fig2 = None
        self.ax2 = None
        self.line_focus = None

        self._build_ui()

        # periodic updates
        self.root.after(50, self._poll_queue)
        self.root.after(250, self._update_plots)
        self.root.after(1000, self._update_metrics_and_biofeedback)

    # ------------------------
    def _build_feature_schema(self) -> List[str]:
        # features for ML:
        # L/R averaged relative band powers + ratios + quality + HRV
        feats = []
        for b in ["delta", "theta", "alpha", "smr", "beta", "gamma"]:
            feats.append(f"rel_{b}_avg")
        feats += [
            "theta_over_smr",
            "alpha_over_beta",
            "beta_over_theta",
            "artifact_rate_avg",
            "flat_rate_avg",
            "HR",
            "RMSSD",
            "SDNN",
        ]
        return feats

    # ------------------------
    def _build_ui(self):
        top = tk.Frame(self.root, bg="#0f1115")
        top.pack(fill="x", padx=14, pady=10)

        tk.Label(top, text="Muse Focus Lab PRO", font=("Segoe UI", 20, "bold"), fg="white", bg="#0f1115").pack(side="left")

        tk.Label(top, text="Session:", font=("Segoe UI", 11), fg="#cfd3da", bg="#0f1115").pack(side="left", padx=(18, 6))
        self.entry_session = ttk.Entry(top, width=20)
        self.entry_session.insert(0, "unknown")
        self.entry_session.pack(side="left")

        tk.Label(top, text="Game:", font=("Segoe UI", 11), fg="#cfd3da", bg="#0f1115").pack(side="left", padx=(16, 6))
        self.cb_game = ttk.Combobox(top, values=["Moving Dot", "Saccade Targets", "Breathing Circle", "Circular Track"], state="readonly", width=18)
        self.cb_game.set("Circular Track")
        self.cb_game.pack(side="left")

        self.btn_start = ttk.Button(top, text="▶ Start", command=self.start_session)
        self.btn_start.pack(side="right", padx=6)
        self.btn_pause = ttk.Button(top, text="⏸ Pause", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side="right", padx=6)
        self.btn_exit = ttk.Button(top, text="Exit", command=self.root.destroy)
        self.btn_exit.pack(side="right", padx=6)

        main = tk.Frame(self.root, bg="#0f1115")
        main.pack(fill="both", expand=True, padx=14, pady=6)

        left = tk.Frame(main, bg="#0f1115")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right = tk.Frame(main, bg="#0f1115")
        right.pack(side="right", fill="y")

        # phase + timer
        self.lbl_phase = tk.Label(left, text="READY", font=("Segoe UI", 26, "bold"), fg="white", bg="#0f1115")
        self.lbl_phase.pack(anchor="w")

        self.lbl_inst = tk.Label(left, text="اضغط Start", font=("Segoe UI", 16), fg="#f1c40f", bg="#0f1115")
        self.lbl_inst.pack(anchor="w", pady=(4, 10))

        timer_row = tk.Frame(left, bg="#0f1115")
        timer_row.pack(anchor="w", pady=(0, 10))

        self.lbl_timer = tk.Label(timer_row, text="00", font=("Consolas", 54, "bold"), fg="#2ecc71", bg="#0f1115")
        self.lbl_timer.pack(side="left")

        self.progress = ttk.Progressbar(timer_row, orient="horizontal", length=460, mode="determinate")
        self.progress.pack(side="left", padx=14)

        # Live EEG plot
        self.fig = Figure(figsize=(7.0, 3.7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Live EEG (processed view, last 10s)")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("uV")

        x = np.arange(self.max_plot_samples)
        for ch in EEG_CH:
            line, = self.ax.plot(x, np.zeros_like(x))
            self.lines[ch] = line
        self.ax.set_xlim(0, self.max_plot_samples)
        self.ax.set_ylim(-150, 150)

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

        # Live focus plot
        self.fig2 = Figure(figsize=(7.0, 2.0), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("Live Focus (ML)")
        self.ax2.set_xlabel("time (samples)")
        self.ax2.set_ylabel("0..100")
        self.line_focus, = self.ax2.plot(np.arange(200), np.zeros(200))
        self.ax2.set_ylim(0, 100)
        self.canvas_focus = FigureCanvasTkAgg(self.fig2, master=left)
        self.canvas_focus.get_tk_widget().pack(fill="x")

        # Right panel: Game + Metrics + Questionnaire + Calibration
        tk.Label(right, text="Biofeedback Game", font=("Segoe UI", 14, "bold"), fg="white", bg="#0f1115").pack(anchor="w")
        self.game_canvas = tk.Canvas(right, width=360, height=240, bg="#05070b", highlightthickness=2, highlightbackground="#34495e")
        self.game_canvas.pack(pady=8)

        tk.Label(right, text="Live Metrics", font=("Segoe UI", 14, "bold"), fg="white", bg="#0f1115").pack(anchor="w", pady=(8, 0))
        self.lbl_conn = tk.Label(right, text="Connection: ❌", font=("Segoe UI", 11), fg="#cfd3da", bg="#0f1115")
        self.lbl_conn.pack(anchor="w", pady=2)

        self.lbl_quality = tk.Label(right, text="Signal Quality: --", font=("Segoe UI", 11), fg="#cfd3da", bg="#0f1115")
        self.lbl_quality.pack(anchor="w", pady=2)

        self.lbl_focus = tk.Label(right, text="Focus (ML): --", font=("Segoe UI", 13, "bold"), fg="#2ecc71", bg="#0f1115")
        self.lbl_focus.pack(anchor="w", pady=4)

        self.lbl_hrv = tk.Label(right, text="HRV: HR -- | RMSSD -- | SDNN --", font=("Segoe UI", 11), fg="#cfd3da", bg="#0f1115")
        self.lbl_hrv.pack(anchor="w", pady=2)

        self.lbl_resp = tk.Label(right, text="Responsiveness: --", font=("Segoe UI", 12, "bold"), fg="#f1c40f", bg="#0f1115")
        self.lbl_resp.pack(anchor="w", pady=6)

        self.lbl_notes = tk.Label(right, text="", font=("Segoe UI", 10), fg="#cfd3da", bg="#0f1115", justify="left")
        self.lbl_notes.pack(anchor="w", pady=4)

        # Questionnaire
        tk.Label(right, text="Quick Questionnaire (1..10)", font=("Segoe UI", 13, "bold"), fg="white", bg="#0f1115").pack(anchor="w", pady=(8, 0))
        self.q_vars: Dict[str, tk.IntVar] = {}
        for q in ["التركيز", "التوتر", "التعب", "الدافعية"]:
            tk.Label(right, text=q, font=("Segoe UI", 10), fg="#cfd3da", bg="#0f1115").pack(anchor="w")
            var = tk.IntVar(value=5)
            s = tk.Scale(right, from_=1, to=10, orient="horizontal", variable=var, length=320, bg="#0f1115",
                         fg="white", highlightthickness=0)
            s.pack(anchor="w")
            self.q_vars[q] = var

        # Calibration buttons
        cal = tk.Frame(right, bg="#0f1115")
        cal.pack(fill="x", pady=8)

        self.btn_add_calib = ttk.Button(cal, text="➕ Add Calibration Point (uses current features + focus rating)", command=self.add_calibration_point)
        self.btn_add_calib.pack(fill="x", pady=2)

        self.btn_train = ttk.Button(cal, text="🧠 Train ML Model", command=self.train_model)
        self.btn_train.pack(fill="x", pady=2)

        self.lbl_model = tk.Label(right, text="ML Model: not trained", font=("Segoe UI", 10), fg="#cfd3da", bg="#0f1115")
        self.lbl_model.pack(anchor="w", pady=4)

        # Status
        self.lbl_status = tk.Label(self.root, text="Ready.", font=("Segoe UI", 10), fg="#cfd3da", bg="#0f1115")
        self.lbl_status.pack(fill="x", padx=14, pady=(0, 10))

        self._create_game()

    # ------------------------
    def _set_status(self, t: str):
        self.lbl_status.config(text=t)

    def toggle_pause(self):
        if not self._running:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.btn_pause.config(text="▶ Resume")
            self._set_status("Paused.")
        else:
            self._pause_event.set()
            self.btn_pause.config(text="⏸ Pause")
            self._set_status("Resumed.")

    # ------------------------
    def _speak_async(self, text: str):
        def worker():
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception as e:
                LOGGER.warning("TTS error: %s", e)
        threading.Thread(target=worker, daemon=True).start()

    # ------------------------
    def _create_game(self):
        choice = self.cb_game.get()
        if choice == "Moving Dot":
            self.game = GameMovingDot(self.game_canvas)
        elif choice == "Saccade Targets":
            self.game = GameSaccadeTargets(self.game_canvas)
        elif choice == "Breathing Circle":
            self.game = GameBreathingCircle(self.game_canvas)
        else:
            self.game = GameCircularTrack(self.game_canvas)

    def _game_start(self):
        if self.game is None:
            self._create_game()
        if self.game:
            self.game.start()

    def _game_stop(self):
        if self.game:
            self.game.stop()

    def _game_tick_loop(self):
        if self.game and self.game.running:
            self.game.tick()
        self.root.after(25, self._game_tick_loop)

    # ------------------------
    # Feature extraction for ML + live scoring
    # ------------------------
    def _extract_live_feature_vector(self) -> Optional[np.ndarray]:
        win = int(FS_EEG * LIVE_FEATURE_WIN_SEC)
        per_ch: Dict[str, Dict[str, float]] = {}

        for ch in EEG_CH:
            data = np.array(self.live_eeg[ch], dtype=float)
            if len(data) < win:
                return None
            seg = data[-win:]
            seg_p = preprocess_eeg(seg, FS_EEG)
            per_ch[ch] = compute_band_features(seg_p, FS_EEG)

        # L/R average rel bands
        feat_map: Dict[str, float] = {}
        for b in ["delta", "theta", "alpha", "smr", "beta", "gamma"]:
            L, R = lr_avg(per_ch, f"rel_{b}")
            feat_map[f"rel_{b}_avg"] = (L + R) / 2.0

        # ratios
        theta = feat_map["rel_theta_avg"]
        alpha = feat_map["rel_alpha_avg"]
        smr = feat_map["rel_smr_avg"]
        beta = feat_map["rel_beta_avg"]

        feat_map["theta_over_smr"] = safe_ratio(theta, smr)
        feat_map["alpha_over_beta"] = safe_ratio(alpha, beta)
        feat_map["beta_over_theta"] = safe_ratio(beta, theta)

        # quality
        art = sum(per_ch[ch]["artifact_flag"] for ch in EEG_CH) / 4.0
        flat = sum(per_ch[ch]["flatline_flag"] for ch in EEG_CH) / 4.0
        feat_map["artifact_rate_avg"] = float(art)
        feat_map["flat_rate_avg"] = float(flat)

        # HRV (optional)
        ppg = np.array(self.live_ppg, dtype=float)
        hrv = compute_hrv_from_ppg(ppg, FS_PPG) if ppg.size >= FS_PPG * 8 else {}
        feat_map["HR"] = float(hrv.get("HR", np.nan))
        feat_map["RMSSD"] = float(hrv.get("RMSSD", np.nan))
        feat_map["SDNN"] = float(hrv.get("SDNN", np.nan))

        # build ordered vector
        vec = np.array([feat_map.get(n, np.nan) for n in self.feature_names], dtype=float)

        # replace nans for ML safety (HRV can be nan)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def _compute_focus_raw(self, feat: np.ndarray) -> float:
        # simple heuristic score 0..100
        # higher (SMR+Beta) vs (Theta+Alpha) = more focused/alert in many tasks
        m = {name: feat[i] for i, name in enumerate(self.feature_names)}
        num = (m["rel_smr_avg"] + m["rel_beta_avg"])
        den = (m["rel_theta_avg"] + m["rel_alpha_avg"])
        raw = safe_ratio(num, den)
        score = 100.0 * (1.0 - math.exp(-2.0 * raw))
        return float(np.clip(score, 0, 100))

    # ------------------------
    # Calibration + ML
    # ------------------------
    def add_calibration_point(self):
        feat = self._extract_live_feature_vector()
        if feat is None:
            messagebox.showwarning("Calibration", "Not enough live EEG data yet. Wait 5 seconds and try again.")
            return

        # Use the questionnaire "التركيز" as label (1..10) for personalization
        y = float(self.q_vars["التركيز"].get())
        self.calib_X.append(feat)
        self.calib_y.append(y)
        self._set_status(f"Added calibration point #{len(self.calib_y)} with focus rating={y:.0f}/10")

    def train_model(self):
        if len(self.calib_y) < 6:
            messagebox.showwarning("ML", "Need at least 6 calibration points (better 10+).")
            return

        X = np.vstack(self.calib_X)
        y = np.array(self.calib_y, dtype=float)

        # Fit ridge
        self.model.fit(X, y)
        self.lbl_model.config(text=f"ML Model: trained (points={len(y)})")
        self._set_status("ML model trained. Biofeedback will follow ML Focus more strongly.")
        LOGGER.info("ML model trained with %d points.", len(y))

    # ------------------------
    # Session controls
    # ------------------------
    def start_session(self):
        if self._running:
            return
        self._running = True
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸ Pause")
        self._pause_event.set()

        self.session_name = self.entry_session.get().strip() or "unknown"
        self._create_game()
        self._game_tick_loop()

        # reset exports
        self.raw_packets_eeg.clear()
        self.raw_samples_eeg.clear()
        self.raw_samples_ppg.clear()
        self.phase_timeline.clear()
        self.focus_history.clear()
        self.current_phase = "READY"

        t = threading.Thread(target=self._ble_thread_runner, daemon=True)
        t.start()

    def _ble_thread_runner(self):
        try:
            asyncio.run(self._ble_protocol())
        except Exception as e:
            LOGGER.exception("BLE thread error: %s", e)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self._finish_ui)

    def _finish_ui(self):
        self._running = False
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self._game_stop()

    # ------------------------
    # BLE protocol
    # ------------------------
    async def _ble_protocol(self):
        self.root.after(0, lambda: self._set_status("Scanning for Muse… (close any app using the headset)"))
        addr = await find_muse_address()
        if not addr:
            raise RuntimeError("Muse not found. Turn it ON and close other apps using it.")

        self.root.after(0, lambda: self.lbl_conn.config(text="Connection: ✅"))
        LOGGER.info("Connecting to: %s", addr)

        # Buffers for packets (phase features later)
        buffers_eeg: Dict[str, List[Tuple[datetime, bytes]]] = {u: [] for u in EEG_UUIDS}
        buffers_ppg: List[Tuple[datetime, bytes]] = []

        def make_eeg_handler(uuid: str):
            def _handler(sender: int, data: bytearray):
                if data is None or len(data) < 20:
                    return
                ts = datetime.now()
                raw20 = bytes(data[:20])
                buffers_eeg[uuid].append((ts, raw20))
                self.raw_packets_eeg.append({
                    "timestamp": ts.isoformat(timespec="milliseconds"),
                    "uuid": uuid,
                    "raw20_hex": raw20.hex()
                })
                # decode samples for live + export
                try:
                    samples = decode_eeg_packet(raw20)
                    for v in samples:
                        self.q.put(("eeg", ts, (uuid, float(v))))
                except Exception as e:
                    LOGGER.debug("EEG decode error: %s", e)
            return _handler

        def ppg_handler(sender: int, data: bytearray):
            if data is None or len(data) == 0:
                return
            ts = datetime.now()
            raw = bytes(data)
            buffers_ppg.append((ts, raw))
            # decode for live hrv
            try:
                vals = decode_ppg_packet(raw)
                for v in vals:
                    self.q.put(("ppg", ts, float(v)))
            except Exception:
                pass

        async with BleakClient(addr, timeout=60.0) as client:
            if not client.is_connected:
                raise RuntimeError("Connect failed.")

            # subscribe EEG
            for u in EEG_UUIDS:
                await client.start_notify(u, make_eeg_handler(u))

            # optional PPG
            ppg_enabled = False
            try:
                await client.start_notify(UUID_PPG, ppg_handler)
                ppg_enabled = True
                LOGGER.info("PPG subscribed.")
            except Exception as e:
                LOGGER.warning("PPG not available (continuing without HRV). %s", e)

            # start stream
            try:
                await client.write_gatt_char(UUID_CONTROL, bytes([0x03, 0x64, 0x0A]), response=False)
                LOGGER.info("Start stream command sent.")
            except Exception as e:
                LOGGER.warning("Could not start stream: %s", e)

            await asyncio.sleep(WARMUP_SEC)

            # Run protocol
            for phase, instruction, dur, color in PHASES:
                self.current_phase = phase
                start_ts = datetime.now()
                self.phase_timeline.append({"phase": phase, "start": start_ts.isoformat(timespec="seconds"), "end": ""})

                self.root.after(0, lambda p=phase, i=instruction: self.lbl_phase.config(text=p))
                self.root.after(0, lambda i=instruction: self.lbl_inst.config(text=i))
                self._speak_async(instruction)

                # game only for eyes-open
                if "افتح" in instruction:
                    self.root.after(0, self._game_start)
                else:
                    self.root.after(0, self._game_stop)

                for remaining in range(dur, 0, -1):
                    # pause/resume gate
                    self._pause_event.wait()

                    self.root.after(0, lambda r=remaining, d=dur, c=color: self._update_timer_ui(r, d, c))
                    await asyncio.sleep(1)

                end_ts = datetime.now()
                self.phase_timeline[-1]["end"] = end_ts.isoformat(timespec="seconds")

            # stop stream
            try:
                await client.write_gatt_char(UUID_CONTROL, bytes([0x03, 0x65, 0x0A]), response=False)
                LOGGER.info("Stop stream command sent.")
            except Exception:
                pass

            for u in EEG_UUIDS:
                try:
                    await client.stop_notify(u)
                except Exception:
                    pass

            if ppg_enabled:
                try:
                    await client.stop_notify(UUID_PPG)
                except Exception:
                    pass

        self.root.after(0, lambda: self._set_status("Processing + exporting (Excel/PDF/Dashboard)…"))
        out_dir = self._export_all(buffers_eeg, buffers_ppg)
        self.root.after(0, lambda: messagebox.showinfo("Done", f"Saved in:\n{out_dir}"))
        self.root.after(0, lambda: self._set_status(f"Done. Saved in {out_dir.name}"))

    def _update_timer_ui(self, remaining: int, total: int, color: str):
        self.lbl_timer.config(text=f"{remaining:02d}", fg=color)
        pct = (total - remaining) / total * 100.0
        self.progress.config(value=pct)

    # ------------------------
    # Queue processing (BLE -> GUI buffers)
    # ------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, ts, payload = self.q.get_nowait()
                if kind == "eeg":
                    uuid, v = payload
                    ch = EEG_CH[EEG_UUIDS.index(uuid)]
                    self.live_eeg[ch].append(float(v))
                    self.live_ts.append(ts)
                    self.raw_samples_eeg.append({
                        "timestamp": ts.isoformat(timespec="milliseconds"),
                        "phase": self.current_phase,
                        "channel": ch,
                        "value_uv": float(v)
                    })
                elif kind == "ppg":
                    v = float(payload)
                    self.live_ppg.append(v)
                    self.raw_samples_ppg.append({
                        "timestamp": ts.isoformat(timespec="milliseconds"),
                        "phase": self.current_phase,
                        "ppg_value": v
                    })
        except queue.Empty:
            pass

        self.root.after(50, self._poll_queue)

    # ------------------------
    # Plots
    # ------------------------
    def _update_plots(self):
        # EEG plot
        for ch in EEG_CH:
            data = np.array(self.live_eeg[ch], dtype=float)
            if len(data) < 10:
                continue
            try:
                y = preprocess_eeg(data, FS_EEG)
            except Exception:
                y = data

            if len(y) < self.max_plot_samples:
                pad = np.zeros(self.max_plot_samples - len(y))
                y2 = np.concatenate([pad, y])
            else:
                y2 = y[-self.max_plot_samples:]

            self.lines[ch].set_ydata(y2)

        self.canvas_plot.draw_idle()

        # focus plot
        fh = [r["focus_ml"] for r in self.focus_history[-200:]] if self.focus_history else []
        if fh:
            y = np.array(fh, dtype=float)
            if len(y) < 200:
                y = np.concatenate([np.zeros(200-len(y)), y])
            self.line_focus.set_ydata(y)
            self.canvas_focus.draw_idle()

        self.root.after(250, self._update_plots)

    # ------------------------
    # Live metrics + Biofeedback (1Hz)
    # ------------------------
    def _update_metrics_and_biofeedback(self):
        feat = self._extract_live_feature_vector()
        if feat is None:
            self.lbl_quality.config(text="Signal Quality: collecting…")
            self.lbl_focus.config(text="Focus (ML): …")
            self.root.after(1000, self._update_metrics_and_biofeedback)
            return

        # quality text
        m = {name: feat[i] for i, name in enumerate(self.feature_names)}
        art = float(m["artifact_rate_avg"])
        flat = float(m["flat_rate_avg"])
        quality = "GOOD"
        if flat > 0.25:
            quality = "BAD (flatline/disconnect?)"
        elif art > 0.50:
            quality = "NOISY (movement/blinks)"
        elif art > 0.25:
            quality = "FAIR"
        self.metrics.quality = quality
        self.metrics.artifact_rate = art
        self.metrics.flat_rate = flat

        # focus heuristic
        focus_raw = self._compute_focus_raw(feat)
        self.metrics.focus_raw = focus_raw

        # ML focus
        if self.model.trained:
            # model predicts 1..10 (roughly). map to 0..100
            yhat = self.model.predict(feat)
            focus_ml = float(np.clip((yhat - 1.0) / 9.0 * 100.0, 0.0, 100.0))
        else:
            focus_ml = focus_raw  # fallback
        self.metrics.focus_ml = focus_ml

        # HRV
        ppg = np.array(self.live_ppg, dtype=float)
        hrv = compute_hrv_from_ppg(ppg, FS_PPG) if ppg.size >= FS_PPG * 8 else {}
        self.metrics.hr = float(hrv.get("HR", np.nan))
        self.metrics.rmssd = float(hrv.get("RMSSD", np.nan))
        self.metrics.sdnn = float(hrv.get("SDNN", np.nan))

        # update labels
        self.lbl_quality.config(text=f"Signal Quality: {quality} | art~{art:.2f} flat~{flat:.2f}")

        # biofeedback: color of focus label
        if focus_ml >= 70:
            self.lbl_focus.config(fg="#2ecc71")
        elif focus_ml >= 45:
            self.lbl_focus.config(fg="#f1c40f")
        else:
            self.lbl_focus.config(fg="#e74c3c")
        self.lbl_focus.config(text=f"Focus (ML): {focus_ml:5.1f}/100  | raw={focus_raw:5.1f}")

        if np.isfinite(self.metrics.hr):
            self.lbl_hrv.config(text=f"HRV: HR {self.metrics.hr:5.1f} | RMSSD {self.metrics.rmssd:6.3f} | SDNN {self.metrics.sdnn:6.3f}")
        else:
            self.lbl_hrv.config(text="HRV: (PPG not available or not enough data)")

        # adjust game speed based on ML focus (true biofeedback)
        # Map focus 0..100 => speed factor 0.7..2.5
        speed = 0.7 + (focus_ml / 100.0) * 1.8
        if self.game:
            self.game.set_speed(speed)

        # save focus history for dashboard/export
        self.focus_history.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "phase": self.current_phase,
            "focus_raw": focus_raw,
            "focus_ml": focus_ml,
            "artifact_rate": art,
            "flat_rate": flat,
            "HR": self.metrics.hr,
            "RMSSD": self.metrics.rmssd,
            "SDNN": self.metrics.sdnn,
        })

        self.lbl_notes.config(text="Biofeedback: speed↑ when Focus↑. For better signal: sit still, relax jaw/forehead, blink less.")

        self.root.after(1000, self._update_metrics_and_biofeedback)

    # ------------------------
    # Export: CSV + Excel + PDF report + phase features
    # ------------------------
    def _export_all(self, buffers_eeg: Dict[str, List[Tuple[datetime, bytes]]], buffers_ppg: List[Tuple[datetime, bytes]]) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"muse_focus_lab_pro_{self.session_name}_{stamp}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save raw sample CSVs
        df_samples = pd.DataFrame(self.raw_samples_eeg)
        df_samples.to_csv(out_dir / "eeg_samples.csv", index=False)

        if self.raw_samples_ppg:
            pd.DataFrame(self.raw_samples_ppg).to_csv(out_dir / "ppg_samples.csv", index=False)

        # Packet-level
        pd.DataFrame(self.raw_packets_eeg).to_csv(out_dir / "eeg_packets.csv", index=False)

        # Timeline
        df_timeline = pd.DataFrame(self.phase_timeline)
        df_timeline.to_csv(out_dir / "phase_timeline.csv", index=False)

        # Focus history
        df_focus = pd.DataFrame(self.focus_history)
        df_focus.to_csv(out_dir / "focus_history.csv", index=False)

        # Questionnaire
        questionnaire = {k: int(v.get()) for k, v in self.q_vars.items()}
        pd.Series(questionnaire).to_json(out_dir / "questionnaire.json", force_ascii=False)

        # Features by phase (OE1/CE/OE2 + include REST for completeness)
        df_phase_features = self._compute_features_by_phase_from_samples(df_samples)
        df_phase_features.to_csv(out_dir / "features_by_phase.csv", index=False)

        # Responsiveness score
        resp = compute_responsiveness(df_phase_features)
        self.lbl_resp.config(text=f"Responsiveness: {resp:5.1f}/100" if np.isfinite(resp) else "Responsiveness: --")

        # Excel export
        xlsx_path = out_dir / "report.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df_samples.to_excel(w, sheet_name="EEG_SAMPLES", index=False)
            if self.raw_samples_ppg:
                pd.DataFrame(self.raw_samples_ppg).to_excel(w, sheet_name="PPG_SAMPLES", index=False)
            pd.DataFrame(self.raw_packets_eeg).to_excel(w, sheet_name="EEG_PACKETS", index=False)
            df_timeline.to_excel(w, sheet_name="PHASE_TIMELINE", index=False)
            df_phase_features.to_excel(w, sheet_name="FEATURES_BY_PHASE", index=False)
            df_focus.to_excel(w, sheet_name="FOCUS_HISTORY", index=False)
            pd.DataFrame([questionnaire]).to_excel(w, sheet_name="QUESTIONNAIRE", index=False)
            pd.DataFrame([{"responsiveness_score": resp}]).to_excel(w, sheet_name="RESPONSIVENESS", index=False)

        # PDF report
        pdf_path = out_dir / "report.pdf"
        self._export_pdf_report(pdf_path, df_phase_features, df_focus, questionnaire, resp)

        # Session meta
        meta = {
            "session_name": self.session_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "fs_eeg": FS_EEG,
            "fs_ppg": FS_PPG,
            "notch_hz": NOTCH_HZ,
            "bandpass": EEG_BANDPASS,
            "bands": BANDS,
            "game": self.cb_game.get(),
            "protocol": [(p[0], p[2]) for p in PHASES],
            "ml_trained": self.model.trained,
            "ml_calib_points": len(self.calib_y),
        }
        pd.Series(meta).to_json(out_dir / "session_info.json", force_ascii=False)

        return out_dir

    def _compute_features_by_phase_from_samples(self, df_samples: pd.DataFrame) -> pd.DataFrame:
        # Compute band features per phase using last FEATURE_BLOCK_SEC seconds worth of data per phase per channel.
        rows = []
        win = int(FS_EEG * FEATURE_BLOCK_SEC)

        for phase in df_samples["phase"].unique():
            dph = df_samples[df_samples["phase"] == phase]
            per_ch: Dict[str, Dict[str, float]] = {}
            for ch in EEG_CH:
                dc = dph[dph["channel"] == ch]
                if len(dc) < win:
                    continue
                x = dc["value_uv"].to_numpy(dtype=float)[-win:]
                x = preprocess_eeg(x, FS_EEG)
                per_ch[ch] = compute_band_features(x, FS_EEG)

            if len(per_ch) < 4:
                continue

            # L/R avg rel
            out = {"phase": phase}
            for b in ["delta", "theta", "alpha", "smr", "beta", "gamma"]:
                L, R = lr_avg(per_ch, f"rel_{b}")
                out[f"rel_{b}_avg"] = (L + R) / 2.0

            art = sum(per_ch[ch]["artifact_flag"] for ch in EEG_CH) / 4.0
            flat = sum(per_ch[ch]["flatline_flag"] for ch in EEG_CH) / 4.0
            out["artifact_rate_avg"] = float(art)
            out["flat_rate_avg"] = float(flat)

            theta = out["rel_theta_avg"]
            alpha = out["rel_alpha_avg"]
            smr = out["rel_smr_avg"]
            beta = out["rel_beta_avg"]

            out["theta_over_smr"] = safe_ratio(theta, smr)
            out["alpha_over_beta"] = safe_ratio(alpha, beta)
            out["beta_over_theta"] = safe_ratio(beta, theta)

            # Add average focus ML over that phase from focus_history
            df_f = pd.DataFrame(self.focus_history)
            if not df_f.empty:
                out["focus_ml_avg"] = float(df_f[df_f["phase"] == phase]["focus_ml"].mean())
                out["focus_ml_std"] = float(df_f[df_f["phase"] == phase]["focus_ml"].std())
            else:
                out["focus_ml_avg"] = float("nan")
                out["focus_ml_std"] = float("nan")

            rows.append(out)

        return pd.DataFrame(rows).sort_values("phase")

    def _export_pdf_report(self, pdf_path: Path, df_phase: pd.DataFrame, df_focus: pd.DataFrame,
                           questionnaire: Dict[str, int], responsiveness: float) -> None:
        c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
        w, h = A4

        y = h - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Muse Focus Lab PRO - Session Report")
        y -= 20
        c.setFont("Helvetica", 11)
        c.drawString(50, y, f"Session: {self.session_name}")
        y -= 15
        c.drawString(50, y, f"Created: {datetime.now().isoformat(timespec='seconds')}")
        y -= 20

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Questionnaire (1..10)")
        y -= 15
        c.setFont("Helvetica", 11)
        for k, v in questionnaire.items():
            c.drawString(60, y, f"- {k}: {v}")
            y -= 13
        y -= 10

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Responsiveness Score: {responsiveness:5.1f}/100" if np.isfinite(responsiveness) else "Responsiveness Score: --")
        y -= 20

        # Phase table (small)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Features by Phase (summary)")
        y -= 15

        c.setFont("Helvetica", 9)
        cols = ["phase", "rel_alpha_avg", "rel_beta_avg", "rel_theta_avg", "theta_over_smr", "focus_ml_avg", "artifact_rate_avg"]
        header = " | ".join(cols)
        c.drawString(50, y, header)
        y -= 12
        c.line(50, y, w - 50, y)
        y -= 12

        for _, row in df_phase.iterrows():
            if y < 120:
                c.showPage()
                y = h - 60
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, "Features by Phase (cont.)")
                y -= 18
                c.setFont("Helvetica", 9)
            vals = []
            for col in cols:
                v = row.get(col, "")
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            c.drawString(50, y, " | ".join(vals))
            y -= 12

        y -= 12
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Focus History (quick stats)")
        y -= 15
        c.setFont("Helvetica", 11)
        if not df_focus.empty:
            c.drawString(60, y, f"Mean Focus(ML): {df_focus['focus_ml'].mean():.2f}")
            y -= 13
            c.drawString(60, y, f"Std  Focus(ML): {df_focus['focus_ml'].std():.2f}")
            y -= 13
            c.drawString(60, y, f"Min/Max Focus: {df_focus['focus_ml'].min():.2f} / {df_focus['focus_ml'].max():.2f}")
            y -= 13
        else:
            c.drawString(60, y, "No focus data recorded.")
            y -= 13

        y -= 10
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(50, y, "Note: Focus(ML) is a personalized experimental estimate based on calibration + EEG features.")
        c.save()

    # ------------------------
    def run(self):
        self.root.mainloop()


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    app = MuseFocusLabProApp()
    app.run()
