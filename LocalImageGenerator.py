# -*- coding: utf-8 -*-
import sys, os, time, json, uuid, random, datetime, urllib.request, urllib.parse
from pathlib import Path
import subprocess, threading, shutil

import requests
import websocket
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QComboBox, QSlider, QSpinBox,
    QProgressBar, QGroupBox, QFormLayout, QSizePolicy, QTabWidget,
    QScrollArea, QFrame, QFileDialog, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

# -- Config persistence ------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "LocalImageGenerator.json")

DEFAULT_CFG = {
    "comfyui_dir": r"D:\VMWare\AI Playground\resources\ComfyUI",
    "comfyui_url": "http://127.0.0.1:8188",
    "image_dir": r"C:\Users\Administrator\Pictures\Wallpapers",
    "video_dir": r"C:\Users\Administrator\Videos\Generated",
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                saved = json.load(f)
            merged = dict(DEFAULT_CFG)
            merged.update(saved)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_CFG)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

APP_CFG = load_config()
os.makedirs(APP_CFG["image_dir"], exist_ok=True)
os.makedirs(APP_CFG["video_dir"], exist_ok=True)

# -- Model Classification ---------------------------------------------------
VIDEO_KEYWORDS = ["ltx", "wan", "cosmos", "kandinsky", "video", "i2v", "t2v", "vace"]

def classify_model(name):
    low = name.lower()
    for kw in VIDEO_KEYWORDS:
        if kw in low:
            return "video"
    return "image"

IMG_TAG = "[IMG] "
VID_TAG = "[VID] "

def tag_model(name):
    return (VID_TAG if classify_model(name) == "video" else IMG_TAG) + name

def untag_model(tagged):
    for prefix in (IMG_TAG, VID_TAG):
        if tagged.startswith(prefix):
            return tagged[len(prefix):]
    return tagged

# -- Resolution Presets ------------------------------------------------------
IMAGE_PRESETS = {
    "512x512 (SD 1.5)": (512, 512),
    "768x768": (768, 768),
    "1024x1024 (SDXL/Flux)": (1024, 1024),
    "1280x720 (HD)": (1280, 720),
    "1920x1080 (Full HD)": (1920, 1080),
    "2560x1440 (QHD)": (2560, 1440),
    "Custom": None,
}
VIDEO_PRESETS = {
    "512x320 (Fast)": (512, 320),
    "768x512": (768, 512),
    "1024x576": (1024, 576),
    "Custom": None,
}

# -- Worker Thread -----------------------------------------------------------
class GenerationWorker(QThread):
    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    image_done = pyqtSignal(str, QImage)
    video_done = pyqtSignal(str)
    error = pyqtSignal(str)


    def __init__(self, mode, prompt, neg, ckpt, w, h, steps, cfg,
                 frames=16, fps=8,
                 comfyui_url="http://127.0.0.1:8188",
                 image_dir=".", video_dir="."):
        super().__init__()
        self.mode = mode
        self.prompt_text = prompt
        self.neg = neg
        self.ckpt = ckpt
        self.w, self.h = w, h
        self.steps = steps
        self.cfg = cfg
        self.frames = frames
        self.fps = fps
        self.cid = str(uuid.uuid4())
        self.comfyui_url = comfyui_url
        self.image_dir = image_dir
        self.video_dir = video_dir

    def _post(self, workflow):
        p = {"prompt": workflow, "client_id": self.cid}
        data = json.dumps(p).encode()
        req = urllib.request.Request(self.comfyui_url + "/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def _history(self, pid):
        with urllib.request.urlopen(self.comfyui_url + "/history/" + pid) as r:
            return json.loads(r.read())

    def _download(self, fn, sub, tp):
        q = urllib.parse.urlencode({"filename": fn, "subfolder": sub, "type": tp})
        with urllib.request.urlopen(self.comfyui_url + "/view?" + q) as r:
            return r.read()

    def _image_workflow(self):
        seed = random.randint(1, 1 << 50)
        return {
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": self.steps, "cfg": self.cfg,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
                "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0]}},
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": self.ckpt}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": self.w, "height": self.h, "batch_size": 1}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": self.prompt_text, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": self.neg, "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage",
                  "inputs": {"filename_prefix": "WallGen", "images": ["8", 0]}},
        }

    def _video_workflow(self):
        seed = random.randint(1, 1 << 50)
        return {
            "4": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": self.ckpt}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": self.prompt_text, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": self.neg, "clip": ["4", 1]}},
            "5": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": self.w, "height": self.h,
                             "batch_size": self.frames}},
            "3": {"class_type": "KSampler", "inputs": {
                "seed": seed, "steps": self.steps, "cfg": self.cfg,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
                "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0]}},
            "8": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "10": {"class_type": "VHS_VideoCombine", "inputs": {
                "images": ["8", 0], "frame_rate": self.fps,
                "loop_count": 0, "filename_prefix": "VidGen",
                "format": "video/h264-mp4",
                "save_output": True, "pingpong": False}},
        }

    def run(self):
        try:
            ws = websocket.WebSocket()
            ws_url = self.comfyui_url.replace("http://", "ws://").replace("https://", "wss://")
            ws.connect(ws_url + "/ws?clientId=" + self.cid)
            self.log.emit("Generating " + ("video" if self.mode == "video" else "image") + "...")

            wf = self._video_workflow() if self.mode == "video" else self._image_workflow()
            pid = self._post(wf)["prompt_id"]

            while True:
                out = ws.recv()
                if isinstance(out, str):
                    msg = json.loads(out)
                    if msg["type"] == "executing":
                        d = msg["data"]
                        if d["node"] is None and d["prompt_id"] == pid:
                            break
                    elif msg["type"] == "progress":
                        d = msg["data"]
                        self.progress.emit(d["value"], d["max"])
            ws.close()

            hist = self._history(pid)[pid]
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if self.mode == "video":
                for nid in hist["outputs"]:
                    node_out = hist["outputs"][nid]
                    if "gifs" in node_out:
                        info = node_out["gifs"][0]
                        raw = self._download(info["filename"], info["subfolder"], info["type"])
                        dst = os.path.join(self.video_dir, "Generated_" + ts + ".mp4")
                        with open(dst, "wb") as f:
                            f.write(raw)
                        self.log.emit("Video saved: " + dst)
                        self.video_done.emit(dst)
                        return
                self.log.emit("Video generated (check ComfyUI output)")
                self.video_done.emit("")
            else:
                for nid in hist["outputs"]:
                    node_out = hist["outputs"][nid]
                    if "images" in node_out:
                        img = node_out["images"][0]
                        raw = self._download(img["filename"], img["subfolder"], img["type"])
                        dst = os.path.join(self.image_dir, "Generated_" + ts + ".png")
                        with open(dst, "wb") as f:
                            f.write(raw)
                        self.log.emit("Image saved: " + dst)
                        self.image_done.emit(dst, QImage.fromData(raw))
                        return
        except Exception as e:
            self.error.emit(str(e))

# -- Stylesheet --------------------------------------------------------------
STYLE = """
QMainWindow { background: #0f0f14; }
QWidget { color: #d4d4dc; font-family: 'Segoe UI'; }
QGroupBox {
    border: 1px solid #2a2a3a; border-radius: 8px; margin-top: 14px;
    font-weight: 600; color: #7c8aff; padding-top: 14px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6366f1, stop:1 #8b5cf6);
    color: white; border: none; border-radius: 6px;
    padding: 10px; font-weight: 700; font-size: 14px;
}
QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #818cf8, stop:1 #a78bfa); }
QPushButton:disabled { background: #2a2a3a; color: #555; }
QComboBox, QSpinBox, QTextEdit, QLineEdit {
    background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 5px;
    padding: 5px; color: #e0e0e8; selection-background-color: #6366f1;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #1a1a24; color: #e0e0e8;
    selection-background-color: #6366f1; }
QSlider::groove:horizontal { background: #2a2a3a; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #6366f1; width: 16px; margin: -5px 0;
    border-radius: 8px; }
QSlider::sub-page:horizontal { background: #6366f1; border-radius: 3px; }
QProgressBar { border: 1px solid #2a2a3a; border-radius: 5px;
    text-align: center; background: #1a1a24; color: white; height: 22px; }
QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 #6366f1, stop:1 #8b5cf6); border-radius: 4px; }
QTabWidget::pane { border: 1px solid #2a2a3a; border-radius: 6px;
    background: #13131a; }
QTabBar::tab { background: #1a1a24; color: #888; padding: 8px 18px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    margin-right: 2px; font-weight: 600; }
QTabBar::tab:selected { background: #6366f1; color: white; }
QTabBar::tab:hover:!selected { background: #2a2a3a; color: #ccc; }
QScrollArea { border: none; background: transparent; }
"""

# -- Main Window -------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Media Generator")
        self.setMinimumSize(1080, 740)
        self.setStyleSheet(STYLE)
        self.all_models = []

        root = QWidget(); self.setCentralWidget(root)
        root_lay = QHBoxLayout(root)
        root_lay.setContentsMargins(12,12,12,12); root_lay.setSpacing(12)

        # -- Left Panel (scrollable) -----------------------------------------
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(380)
        left = QWidget(); self.left_lay = QVBoxLayout(left)
        self.left_lay.setContentsMargins(0,0,6,0)
        scroll.setWidget(left)

        hdr = QLabel("AI Media Generator")
        hdr.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        hdr.setStyleSheet("color:#8b8fff;")
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_lay.addWidget(hdr)

        self.status = QLabel("Checking ComfyUI...")
        self.status.setStyleSheet("color:#ffaa00; font-size:12px;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_lay.addWidget(self.status)

        self.tabs = QTabWidget()

        # -- IMAGE TAB -------------------------------------------------------
        img_tab = QWidget(); il = QVBoxLayout(img_tab)

        mg = QGroupBox("Model"); ml = QVBoxLayout(mg)
        self.img_model = QComboBox(); ml.addWidget(self.img_model)
        il.addWidget(mg)

        rg = QGroupBox("Resolution"); rl = QVBoxLayout(rg)
        self.img_preset = QComboBox()
        self.img_preset.addItems(IMAGE_PRESETS.keys())
        self.img_preset.setCurrentIndex(2)
        self.img_preset.currentTextChanged.connect(self._img_preset_changed)
        rl.addWidget(self.img_preset)
        cw = QWidget(); cl = QHBoxLayout(cw); cl.setContentsMargins(0,4,0,0)
        self.img_w = QSpinBox(); self.img_w.setRange(64,4096); self.img_w.setSingleStep(64); self.img_w.setValue(1024)
        self.img_h = QSpinBox(); self.img_h.setRange(64,4096); self.img_h.setSingleStep(64); self.img_h.setValue(1024)
        cl.addWidget(QLabel("W:")); cl.addWidget(self.img_w)
        cl.addWidget(QLabel("H:")); cl.addWidget(self.img_h)
        rl.addWidget(cw); self.img_custom_row = cw
        self._img_preset_changed(self.img_preset.currentText())
        il.addWidget(rg)

        pg = QGroupBox("Parameters"); pl = QFormLayout(pg)
        self.img_steps = QSlider(Qt.Orientation.Horizontal); self.img_steps.setRange(1,50); self.img_steps.setValue(20)
        self.img_steps_lbl = QLabel("20")
        self.img_steps.valueChanged.connect(lambda v: self.img_steps_lbl.setText(str(v)))
        pl.addRow("Steps:", self.img_steps_lbl); pl.addRow(self.img_steps)
        self.img_cfg = QSlider(Qt.Orientation.Horizontal); self.img_cfg.setRange(10,200); self.img_cfg.setValue(80)
        self.img_cfg_lbl = QLabel("8.0")
        self.img_cfg.valueChanged.connect(lambda v: self.img_cfg_lbl.setText(str(round(v/10,1))))
        pl.addRow("CFG:", self.img_cfg_lbl); pl.addRow(self.img_cfg)
        il.addWidget(pg)
        il.addStretch()
        self.tabs.addTab(img_tab, "Image")

        # -- VIDEO TAB -------------------------------------------------------
        vid_tab = QWidget(); vl = QVBoxLayout(vid_tab)

        vmg = QGroupBox("Model"); vml = QVBoxLayout(vmg)
        self.vid_model = QComboBox(); vml.addWidget(self.vid_model)
        vl.addWidget(vmg)

        vrg = QGroupBox("Resolution"); vrl = QVBoxLayout(vrg)
        self.vid_preset = QComboBox()
        self.vid_preset.addItems(VIDEO_PRESETS.keys())
        self.vid_preset.setCurrentIndex(1)
        self.vid_preset.currentTextChanged.connect(self._vid_preset_changed)
        vrl.addWidget(self.vid_preset)
        vcw = QWidget(); vcl = QHBoxLayout(vcw); vcl.setContentsMargins(0,4,0,0)
        self.vid_w = QSpinBox(); self.vid_w.setRange(64,2048); self.vid_w.setSingleStep(64); self.vid_w.setValue(768)
        self.vid_h = QSpinBox(); self.vid_h.setRange(64,2048); self.vid_h.setSingleStep(64); self.vid_h.setValue(512)
        vcl.addWidget(QLabel("W:")); vcl.addWidget(self.vid_w)
        vcl.addWidget(QLabel("H:")); vcl.addWidget(self.vid_h)
        vrl.addWidget(vcw); self.vid_custom_row = vcw
        self._vid_preset_changed(self.vid_preset.currentText())
        vl.addWidget(vrg)

        vpg = QGroupBox("Video Settings"); vpl = QFormLayout(vpg)
        self.vid_frames = QSpinBox(); self.vid_frames.setRange(4,128); self.vid_frames.setValue(16)
        vpl.addRow("Frames:", self.vid_frames)
        self.vid_fps = QSpinBox(); self.vid_fps.setRange(1,60); self.vid_fps.setValue(8)
        vpl.addRow("FPS:", self.vid_fps)
        self.vid_steps = QSlider(Qt.Orientation.Horizontal); self.vid_steps.setRange(1,50); self.vid_steps.setValue(20)
        self.vid_steps_lbl = QLabel("20")
        self.vid_steps.valueChanged.connect(lambda v: self.vid_steps_lbl.setText(str(v)))
        vpl.addRow("Steps:", self.vid_steps_lbl); vpl.addRow(self.vid_steps)
        self.vid_cfg = QSlider(Qt.Orientation.Horizontal); self.vid_cfg.setRange(10,200); self.vid_cfg.setValue(80)
        self.vid_cfg_lbl = QLabel("8.0")
        self.vid_cfg.valueChanged.connect(lambda v: self.vid_cfg_lbl.setText(str(round(v/10,1))))
        vpl.addRow("CFG:", self.vid_cfg_lbl); vpl.addRow(self.vid_cfg)
        vl.addWidget(vpg)
        vl.addStretch()
        self.tabs.addTab(vid_tab, "Video")

        # -- SETTINGS TAB ----------------------------------------------------
        set_tab = QWidget(); sl = QVBoxLayout(set_tab)

        dir_group = QGroupBox("Directories"); dl = QFormLayout(dir_group)

        self.comfyui_dir_input = QLineEdit(APP_CFG["comfyui_dir"])
        comfyui_browse = QPushButton("..."); comfyui_browse.setFixedWidth(36)
        comfyui_browse.clicked.connect(lambda: self._browse_dir(self.comfyui_dir_input))
        cr = QHBoxLayout(); cr.addWidget(self.comfyui_dir_input); cr.addWidget(comfyui_browse)
        cw2 = QWidget(); cw2.setLayout(cr)
        dl.addRow("ComfyUI Dir:", cw2)

        self.comfyui_url_input = QLineEdit(APP_CFG["comfyui_url"])
        dl.addRow("ComfyUI URL:", self.comfyui_url_input)

        self.img_dir_input = QLineEdit(APP_CFG["image_dir"])
        img_browse = QPushButton("..."); img_browse.setFixedWidth(36)
        img_browse.clicked.connect(lambda: self._browse_dir(self.img_dir_input))
        ir = QHBoxLayout(); ir.addWidget(self.img_dir_input); ir.addWidget(img_browse)
        iw = QWidget(); iw.setLayout(ir)
        dl.addRow("Image Output:", iw)

        self.vid_dir_input = QLineEdit(APP_CFG["video_dir"])
        vid_browse = QPushButton("..."); vid_browse.setFixedWidth(36)
        vid_browse.clicked.connect(lambda: self._browse_dir(self.vid_dir_input))
        vr = QHBoxLayout(); vr.addWidget(self.vid_dir_input); vr.addWidget(vid_browse)
        vw = QWidget(); vw.setLayout(vr)
        dl.addRow("Video Output:", vw)

        sl.addWidget(dir_group)

        save_cfg_btn = QPushButton("Save Settings")
        save_cfg_btn.clicked.connect(self._save_settings)
        sl.addWidget(save_cfg_btn)
        sl.addStretch()
        self.tabs.addTab(set_tab, "Settings")

        self.left_lay.addWidget(self.tabs)

        # -- Prompts (shared) ------------------------------------------------
        prg = QGroupBox("Prompts"); prl = QVBoxLayout(prg)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Describe what you want...")
        self.prompt.setFixedHeight(90)
        prl.addWidget(QLabel("Positive:")); prl.addWidget(self.prompt)
        self.neg_prompt = QTextEdit()
        self.neg_prompt.setPlaceholderText("What to avoid...")
        self.neg_prompt.setFixedHeight(60)
        self.neg_prompt.setText("text, watermark, ugly, blurry, low quality")
        prl.addWidget(QLabel("Negative:")); prl.addWidget(self.neg_prompt)
        self.left_lay.addWidget(prg)

        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        self.left_lay.addWidget(self.progress_bar)
        self.gen_btn = QPushButton("Generate Image"); self.gen_btn.setFixedHeight(48)
        self.gen_btn.clicked.connect(self.start_gen); self.gen_btn.setEnabled(False)
        self.left_lay.addWidget(self.gen_btn)

        self.tabs.currentChanged.connect(self._tab_changed)

        root_lay.addWidget(scroll)

        # -- Right Panel (Preview) -------------------------------------------
        right = QWidget(); right_lay = QVBoxLayout(right)
        self.preview = QLabel("Output will appear here")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#0a0a10; border:1px solid #2a2a3a; border-radius:8px;")
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_lay.addWidget(self.preview)
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color:#666; font-size:11px;")
        self.path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_lay.addWidget(self.path_label)
        root_lay.addWidget(right)

        threading.Thread(target=self._boot_backend, daemon=True).start()

    # -- Helpers -------------------------------------------------------------
    def _img_preset_changed(self, text):
        custom = (text == "Custom")
        self.img_custom_row.setVisible(custom)
        if not custom and IMAGE_PRESETS.get(text):
            w, h = IMAGE_PRESETS[text]
            self.img_w.setValue(w); self.img_h.setValue(h)

    def _vid_preset_changed(self, text):
        custom = (text == "Custom")
        self.vid_custom_row.setVisible(custom)
        if not custom and VIDEO_PRESETS.get(text):
            w, h = VIDEO_PRESETS[text]
            self.vid_w.setValue(w); self.vid_h.setValue(h)

    def _tab_changed(self, idx):
        if hasattr(self, 'gen_btn'):
            # 0=Image, 1=Video, 2=Settings
            self.gen_btn.setText("Generate Video" if idx == 1 else "Generate Image")

    def _browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text())
        if d:
            line_edit.setText(d)

    def _save_settings(self):
        global APP_CFG
        APP_CFG["comfyui_dir"] = self.comfyui_dir_input.text()
        APP_CFG["comfyui_url"] = self.comfyui_url_input.text()
        APP_CFG["image_dir"] = self.img_dir_input.text()
        APP_CFG["video_dir"] = self.vid_dir_input.text()
        os.makedirs(APP_CFG["image_dir"], exist_ok=True)
        os.makedirs(APP_CFG["video_dir"], exist_ok=True)
        save_config(APP_CFG)
        self.status.setText("Settings saved")
        self.status.setStyleSheet("color:#4ade80; font-size:12px;")

    def _boot_backend(self):
        url = APP_CFG["comfyui_url"]
        try:
            if requests.get(url, timeout=1).status_code == 200:
                self._load_models(); return
        except: pass
        self.status.setText("Starting ComfyUI (GPU)...")
        cdir = APP_CFG["comfyui_dir"]
        cpython = os.path.join(cdir, ".venv", "Scripts", "python.exe")
        subprocess.Popen(
            [cpython, "main.py", "--highvram", "--force-fp16"],
            cwd=cdir, creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(90):
            try:
                if requests.get(url, timeout=1).status_code == 200:
                    self._load_models(); return
            except: time.sleep(1)
        self.status.setText("Failed to start ComfyUI.")

    def _load_models(self):
        url = APP_CFG["comfyui_url"]
        try:
            data = requests.get(url + "/object_info").json()
            raw = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            self.all_models = list(raw)
            imgs = [tag_model(m) for m in raw if classify_model(m) == "image"]
            vids = [tag_model(m) for m in raw if classify_model(m) == "video"]
            if not vids: vids = [tag_model(m) for m in raw]
            if not imgs: imgs = [tag_model(m) for m in raw]
            self.img_model.clear(); self.img_model.addItems(imgs)
            self.vid_model.clear(); self.vid_model.addItems(vids)
            self.gen_btn.setEnabled(True)
            self.status.setText("Ready")
            self.status.setStyleSheet("color:#4ade80; font-size:12px;")
        except Exception as e:
            self.status.setText("Model load error"); print(e)

    def start_gen(self):
        p = self.prompt.toPlainText().strip()
        if not p: return
        self.gen_btn.setEnabled(False); self.progress_bar.setValue(0)
        is_video = self.tabs.currentIndex() == 1
        if is_video:
            ckpt = untag_model(self.vid_model.currentText())
            w, h = self.vid_w.value(), self.vid_h.value()
            steps = self.vid_steps.value(); cfg = self.vid_cfg.value() / 10.0
            frames = self.vid_frames.value(); fps = self.vid_fps.value()
        else:
            ckpt = untag_model(self.img_model.currentText())
            w, h = self.img_w.value(), self.img_h.value()
            steps = self.img_steps.value(); cfg = self.img_cfg.value() / 10.0
            frames, fps = 1, 8

        self.worker = GenerationWorker(
            "video" if is_video else "image",
            p, self.neg_prompt.toPlainText().strip(),
            ckpt, w, h, steps, cfg, frames, fps,
            comfyui_url=APP_CFG["comfyui_url"],
            image_dir=APP_CFG["image_dir"],
            video_dir=APP_CFG["video_dir"])
        self.worker.progress.connect(self._prog)
        self.worker.log.connect(lambda m: self.status.setText(m))
        self.worker.image_done.connect(self._show_img)
        self.worker.video_done.connect(self._show_vid)
        self.worker.error.connect(self._err)
        self.worker.finished.connect(lambda: self.gen_btn.setEnabled(True))
        self.worker.start()

    def _prog(self, cur, mx):
        if mx > 0: self.progress_bar.setValue(int(cur / mx * 100))

    def _show_img(self, path, qimg):
        px = QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(px)
        self.path_label.setText(path)

    def _show_vid(self, path):
        self.preview.setText("Video saved!\n" + path if path else "Video generated.")
        self.path_label.setText(path)

    def _err(self, e):
        self.status.setText("Error: " + e)
        self.status.setStyleSheet("color:#f87171; font-size:12px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
