#!/usr/bin/env python3
"""
Video Quality Enhancer — 4K Upscaler & FPS Booster
Upscales video to 4K resolution and/or increases frame rate using AI.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn
)

console = Console()


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_ffprobe() -> bool:
    """Check if FFprobe is installed."""
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_info(video_path: str) -> dict:
    """Extract video metadata using FFprobe."""
    if not check_ffprobe():
        console.print("[red]❌ FFprobe not found. Install FFmpeg first.[/red]")
        sys.exit(1)

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        video_stream = None
        audio_stream = None
        for stream in data.get("streams", []):
            if stream["codec_type"] == "video" and video_stream is None:
                video_stream = stream
            elif stream["codec_type"] == "audio" and audio_stream is None:
                audio_stream = stream

        if video_stream is None:
            raise ValueError("No video stream found")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        # Parse frame rate (might be "30000/1001" style)
        fps_str = video_stream.get("r_frame_rate", "0")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 0
        else:
            fps = float(fps_str)

        total_frames = int(video_stream.get("nb_frames", 0))
        duration = float(data.get("format", {}).get("duration", 0))
        codec = video_stream.get("codec_name", "unknown")
        bitrate = data.get("format", {}).get("bit_rate", "N/A")

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "codec": codec,
            "bitrate": bitrate,
            "has_audio": audio_stream is not None,
        }
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ FFprobe error: {e.stderr}[/red]")
        sys.exit(1)


def display_video_info(video_path: str):
    """Display video information in a formatted table."""
    info = get_video_info(video_path)

    table = Table(title=f"📹 {Path(video_path).name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Resolution", f"{info['width']}×{info['height']}")
    table.add_row("Frame Rate", f"{info['fps']:.2f} fps")
    table.add_row("Duration", f"{info['duration']:.2f}s ({info['duration'] / 60:.1f} min)")
    table.add_row("Codec", info["codec"])
    table.add_row("Bitrate", info["bitrate"])
    table.add_row("Audio", "✅ Yes" if info["has_audio"] else "❌ No")

    if info["total_frames"] > 0:
        table.add_row("Total Frames", str(info["total_frames"]))

    # Suggested upscale target
    table.add_row(
        "Suggested 4K",
        "✅ Already 4K" if info["width"] >= 3840 else "🚀 Can upscale"
    )

    console.print(table)
    return info


def extract_audio(input_path: str, output_path: str) -> bool:
    """Extract audio from video to a temporary AAC file."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn", "-acodec", "aac",
        "-y", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def upscale_frame_opencv(frame: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Upscale a single frame using OpenCV's super-resolution (EDSR).
    Falls back to Lanczos interpolation if SR model unavailable.
    """
    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    # Try to use DNN super-resolution if available
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = os.path.join(os.path.dirname(__file__), "models", "EDSR_x2.pb")
        if os.path.exists(model_path):
            sr.readModel(model_path)
            sr.setModel("edsr", 2)
            upscaled = sr.upsample(frame)
            return upscaled
    except Exception:
        pass

    # Fallback: Lanczos interpolation (high-quality)
    upscaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return upscaled


def upscale_ffmpeg(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: Optional[float] = None,
    hwaccel: bool = False,
    quality: int = 23,
) -> bool:
    """
    Upscale video using FFmpeg with high-quality settings.
    Uses Lanczos scaling for best non-AI quality.
    """
    # Build filter chain
    filter_parts = []

    # Scale to target resolution using Lanczos
    filter_parts.append(f"scale={width}:{height}:flags=lanczos")

    # Apply sharpening to enhance details after upscaling
    filter_parts.append("unsharp=3:3:0.5:3:3:0.0")

    # FPS adjustment if needed
    if fps:
        filter_parts.append(f"fps={fps}")

    filter_chain = ",".join(filter_parts)

    # Encoding settings
    if hwaccel:
        # Try NVIDIA NVENC
        encoder = "h264_nvenc"
        encoder_opts = [
            "-preset", "p7",
            "-rc", "vbr_hq",
            "-cq", str(quality),
            "-b:v", "50M",
            "-maxrate", "80M",
        ]
    else:
        encoder = "libx264"
        encoder_opts = [
            "-preset", "slow",
            "-crf", str(quality),
        ]

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", filter_chain,
        "-c:v", encoder,
        *encoder_opts,
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    console.print(f"[cyan]⚡ Running FFmpeg upscale: {width}×{height}[/cyan]")

    process = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        universal_newlines=True
    )

    # Show progress by parsing FFmpeg output
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Upscaling...", total=100)

        for line in process.stderr:
            if "time=" in line:
                # Parse time from FFmpeg output
                try:
                    time_str = line.split("time=")[1].split()[0]
                    h, m, s = time_str.split(":")
                    seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    # Get total duration from input info
                    info = get_video_info(input_path)
                    if info["duration"] > 0:
                        pct = min(100, (seconds / info["duration"]) * 100)
                        progress.update(task, completed=pct)
                except (ValueError, IndexError):
                    progress.update(task, advance=0.1)

        process.wait()

    if process.returncode != 0:
        console.print(f"[red]❌ FFmpeg failed with code {process.returncode}[/red]")
        return False

    return True


def interpolate_frames_ffmpeg(
    input_path: str,
    output_path: str,
    target_fps: float,
    hwaccel: bool = False,
) -> bool:
    """
    Increase frame rate using FFmpeg motion-interpolated filter (minterpolate).
    This creates smooth slow-motion or high-FPS video using motion compensation.
    """
    # Get current FPS
    info = get_video_info(input_path)
    current_fps = info["fps"]

    if target_fps <= current_fps:
        console.print(
            f"[yellow]⚠ Target FPS ({target_fps}) not higher than current ({current_fps:.1f}). "
            f"Just re-encoding at same FPS.[/yellow]"
        )
        target_fps = current_fps

    # Build filter: motion-compensated frame interpolation
    # mi_mode=mci: Motion-compensated interpolation
    # me_mode=bidir: Bidirectional motion estimation (better quality)
    # mc_mode=aobmc: Adaptive overlapped block motion compensation
    filter_chain = (
        f"minterpolate="
        f"mi_mode=mci:"
        f"me_mode=bidir:"
        f"mc_mode=aobmc:"
        f"vsbmc=1:"
        f"fps={target_fps}"
    )

    if hwaccel:
        encoder = "h264_nvenc"
        encoder_opts = ["-preset", "p7", "-rc", "vbr_hq", "-cq", "23", "-b:v", "50M"]
    else:
        encoder = "libx264"
        encoder_opts = ["-preset", "slow", "-crf", "18"]

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", filter_chain,
        "-c:v", encoder,
        *encoder_opts,
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    console.print(f"[cyan]⚡ Interpolating frames: {current_fps:.1f} → {target_fps} fps[/cyan]")

    process = subprocess.Popen(
        cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
        universal_newlines=True
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Interpolating...", total=100)
        duration = info["duration"]

        for line in process.stderr:
            if "time=" in line:
                try:
                    time_str = line.split("time=")[1].split()[0]
                    h, m, s = time_str.split(":")
                    seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    if duration > 0:
                        pct = min(100, (seconds / duration) * 100)
                        progress.update(task, completed=pct)
                except (ValueError, IndexError):
                    progress.update(task, advance=0.1)

        process.wait()

    if process.returncode != 0:
        console.print(f"[red]❌ Frame interpolation failed[/red]")
        return False

    return True


def enhance_video(
    input_path: str,
    output_path: str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    target_fps: Optional[float] = None,
    to_4k: bool = False,
    hwaccel: bool = False,
    quality: int = 23,
) -> bool:
    """
    Main enhancement pipeline. Can upscale, increase FPS, or both.
    Uses FFmpeg for reliable multi-stage processing.
    """
    if not check_ffmpeg():
        console.print("[red]❌ FFmpeg not found. Install it first: sudo apt install ffmpeg[/red]")
        return False

    input_path = os.path.abspath(input_path)
    if not os.path.exists(input_path):
        console.print(f"[red]❌ Input file not found: {input_path}[/red]")
        return False

    info = get_video_info(input_path)
    console.print(f"\n[bold]📹 Input:[/bold] {Path(input_path).name}")
    console.print(f"   Resolution: {info['width']}×{info['height']}")
    console.print(f"   FPS: {info['fps']:.2f}")
    console.print(f"   Duration: {info['duration']:.1f}s\n")

    # Determine target resolution
    if to_4k:
        target_width = 3840
        target_height = 2160
    elif target_width is None and target_height is None:
        target_width = info["width"]
        target_height = info["height"]
    elif target_width is None:
        # Maintain aspect ratio
        ratio = target_height / info["height"]
        target_width = int(info["width"] * ratio)
    elif target_height is None:
        ratio = target_width / info["width"]
        target_height = int(info["height"] * ratio)

    # Ensure dimensions are even (required by most codecs)
    target_width = target_width + (target_width % 2)
    target_height = target_height + (target_height % 2)

    needs_upscale = (target_width > info["width"] or target_height > info["height"])
    needs_fps = (target_fps is not None and target_fps > info["fps"])

    if not needs_upscale and not needs_fps:
        console.print("[yellow]⚠ No enhancement needed. Output would be identical to input.[/yellow]")
        if not target_fps:
            console.print("   Use --fps to increase frame rate or --to-4k to upscale.")
        return False

    tmp_dir = tempfile.mkdtemp(prefix="video_enhance_")

    try:
        # Strategy: do upscale first, then FPS boost (better quality)
        current_input = input_path

        if needs_upscale:
            upscaled_path = os.path.join(tmp_dir, "upscaled.mp4")
            console.print(f"\n[bold]🚀 Phase 1: Upscaling to {target_width}×{target_height}[/bold]")
            ok = upscale_ffmpeg(
                current_input, upscaled_path,
                target_width, target_height,
                fps=None,  # Don't change FPS yet
                hwaccel=hwaccel,
                quality=quality,
            )
            if not ok:
                return False
            current_input = upscaled_path

        if needs_fps:
            fps_path = os.path.join(tmp_dir, "interpolated.mp4")
            console.print(f"\n[bold]⚡ Phase 2: Boosting FPS to {target_fps}[/bold]")
            ok = interpolate_frames_ffmpeg(
                current_input, fps_path,
                target_fps,
                hwaccel=hwaccel,
            )
            if not ok:
                return False
            current_input = fps_path

        # Copy final result to output
        shutil.copy2(current_input, output_path)

        # Show result info
        out_info = get_video_info(output_path)
        console.print(f"\n[bold green]✅ Enhancement complete![/bold green]")
        console.print(f"   Output: {output_path}")
        console.print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        console.print(f"   Resolution: {out_info['width']}×{out_info['height']}")
        console.print(f"   FPS: {out_info['fps']:.2f}")
        console.print(f"   Duration: {out_info['duration']:.1f}s")
        console.print(f"   Audio: {'✅ Preserved' if out_info['has_audio'] else '❌ None'}")

        if info["duration"] > 0 and out_info["duration"] > 0:
            ratio = out_info["duration"] / info["duration"]
            if needs_fps and abs(ratio - 1.0) > 0.01:
                console.print(f"   ⏱ Duration changed: ×{ratio:.2f} (expected for FPS change)")

        return True

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary files
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def batch_process(
    input_dir: str,
    output_dir: str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    target_fps: Optional[float] = None,
    to_4k: bool = False,
    hwaccel: bool = False,
):
    """Process all video files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    video_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in video_extensions and f.is_file()
    ]

    if not video_files:
        console.print(f"[yellow]⚠ No video files found in {input_dir}[/yellow]")
        return

    console.print(f"[bold]📁 Found {len(video_files)} video(s) to process[/bold]\n")

    successful = 0
    failed = 0

    for video_file in sorted(video_files):
        out_file = output_path / f"{video_file.stem}_enhanced{video_file.suffix}"
        console.print(
            f"\n[bold]{'='*60}[/bold]\n"
            f"[bold]Processing:[/bold] {video_file.name}"
        )

        ok = enhance_video(
            str(video_file),
            str(out_file),
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            to_4k=to_4k,
            hwaccel=hwaccel,
        )

        if ok:
            successful += 1
        else:
            failed += 1

    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]📊 Batch Complete: {successful} succeeded, {failed} failed[/bold]")


def main():
    parser = argparse.ArgumentParser(
        description="🎬 Video Quality Enhancer — Upscale to 4K & Boost FPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.mp4 -o output.mp4 --to-4k
  %(prog)s input.mp4 -o output.mp4 --to-4k --fps 60
  %(prog)s input.mp4 -o output.mp4 --width 3840 --height 2160 --fps 120
  %(prog)s input.mp4 --info
  %(prog)s ./videos/ -o ./enhanced/ --to-4k --fps 60
        """,
    )

    parser.add_argument("input", help="Input video file or directory (for batch)")
    parser.add_argument("-o", "--output", default=None, help="Output file or directory")

    quality = parser.add_argument_group("Quality Settings")
    quality.add_argument("--to-4k", action="store_true", help="Upscale to 4K (3840×2160)")
    quality.add_argument("--width", type=int, default=None, help="Target width")
    quality.add_argument("--height", type=int, default=None, help="Target height")
    quality.add_argument("--fps", type=float, default=None, help="Target frame rate")
    quality.add_argument("--quality", type=int, default=23, help="Video quality (lower = better, 0-51, default: 23)")
    quality.add_argument("--hwaccel", action="store_true", help="Use NVIDIA GPU hardware acceleration")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("--info", action="store_true", help="Show video info and exit")

    args = parser.parse_args()

    # Check if input is a directory (batch mode)
    if os.path.isdir(args.input):
        output_dir = args.output or f"{args.input}_enhanced"
        console.print(f"[bold]📁 Batch mode: {args.input} → {output_dir}[/bold]")
        batch_process(
            args.input, output_dir,
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps,
            to_4k=args.to_4k,
            hwaccel=args.hwaccel,
        )
        return

    # Single file mode
    if args.info:
        display_video_info(args.input)
        return

    # Auto-generate output name if not provided
    if args.output is None:
        input_path = Path(args.input)
        output_name = f"{input_path.stem}_enhanced{input_path.suffix}"
        args.output = str(input_path.parent / output_name)

    enhance_video(
        args.input, args.output,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps,
        to_4k=args.to_4k,
        hwaccel=args.hwaccel,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
