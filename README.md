# AI Media Generator

A standalone PyQt6 desktop application for generating AI images and videos using a local ComfyUI backend.

## Features
- **Image Generation** with resolution presets (512x512 to 2560x1440) and custom dimensions
- **Video Generation** with configurable frame count, FPS, and resolution
- **Model Auto-Classification** - models tagged as `[IMG]` or `[VID]` automatically
- **GPU Accelerated** - launches ComfyUI with `--highvram --force-fp16` for Intel Arc / NVIDIA GPUs
- **Persistent Settings** - configurable directories for ComfyUI, image output, and video output
- **Auto Backend Launch** - silently starts ComfyUI if not already running

## Requirements
- Python 3.11+
- Local ComfyUI installation
- PyQt6, requests, websocket-client

## Install
```bash
pip install -r requirements.txt
```

## Usage
```bash
python LocalImageGenerator.py
```

## Configuration
On first run, edit the **Settings** tab to point to your ComfyUI directory and output folders. Settings are saved to `LocalImageGenerator.json`.
