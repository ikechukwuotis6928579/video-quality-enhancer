# Video Quality Enhancer — 4K Upscaler & FPS Booster

AI-powered video enhancement tool that upscales any video to **4K resolution** and increases **frame rate** for buttery-smooth playback.

## Features

- **4K Upscaling** — Real-ESRGAN AI model for intelligent upscaling (preserves details, removes artifacts)
- **FPS Boost** — Frame interpolation using FFmpeg motion estimation (smooth 30→60, 60→120, etc.)
- **Batch Processing** — Process multiple videos at once
- **Preserve Audio** — Original audio is untouched
- **Progress Tracking** — Real-time progress bar
- **Hardware Acceleration** — Auto-detects NVIDIA GPU (CUDA/NVENC) for faster encoding

## Requirements

- Python 3.8+
- FFmpeg installed on your system
- 4GB+ RAM (8GB+ recommended for 4K)
- NVIDIA GPU recommended for faster processing (optional)

## Installation

```bash
# 1. Install FFmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg -y
# macOS:
brew install ffmpeg
# Windows: Download from https://ffmpeg.org/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download AI models (optional, for better upscaling)
python download_models.py
```

## Usage

```bash
# Basic upscale to 4K at original FPS
python enhance.py input.mp4 -o output.mp4 --to-4k

# Upscale to 4K + boost FPS to 60
python enhance.py input.mp4 -o output.mp4 --to-4k --fps 60

# Custom resolution + FPS
python enhance.py input.mp4 -o output.mp4 --width 3840 --height 2160 --fps 120

# Boost FPS only (no upscaling)
python enhance.py input.mp4 -o output.mp4 --fps 60

# Batch process all videos in a folder
python enhance.py ./videos/ -o ./enhanced/ --to-4k --fps 60

# Use hardware acceleration (NVIDIA)
python enhance.py input.mp4 -o output.mp4 --to-4k --fps 60 --hwaccel

# Preview info without processing
python enhance.py input.mp4 --info
```

## How It Works

### 4K Upscaling
1. Extracts frames from the video
2. Passes each frame through Real-ESRGAN (AI super-resolution)
3. Rebuilds the video at 4K resolution
4. Preserves original audio track

### FPS Boosting
1. Analyzes motion between consecutive frames using optical flow
2. Generates intermediate frames intelligently
3. Produces smooth, high-frame-rate video without the "soap opera" look

## Examples

| Before | After |
|--------|-------|
| 720p @ 30fps | 4K @ 60fps |
| 1080p @ 24fps | 4K @ 120fps |
| 480p @ 30fps | 1080p @ 60fps |

## License

MIT
