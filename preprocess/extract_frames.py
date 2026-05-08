"""
Dashcam Frame Extraction + Indexing

Features:
- Processes all .mp4 files in a folder
- Extracts N frames per second with ffmpeg (default: 1 fps)
- Configurable output width (default: 1280px)
- Parses YYYYMMDDHHMMSS timestamp from filename
- Computes real wall-clock time for each frame
- Reads city info AND coordinates from route_changes.csv
  and labels each frame as daylight / twilight / night
- Optionally skips videos that fall entirely within daylight (MVP mode)
- Writes everything to a single frames_index.csv

route_changes.csv format (with header, 4 columns):
    start_time,city,lat,lon
    2000-01-01T00:00:00,Antalya,36.8969,30.7133
    2025-08-19T00:00:00,Erzurum,39.9000,41.2700
    2025-08-19T13:00:00,Karaman,37.1759,33.2287

The first row acts as the default fallback (use a very old date so it
matches any video earlier than your real route entries).

The format is "from this time onwards, I was in <city> at <lat,lon>".
The next row's start_time acts as an implicit end_time for the current
location.

Usage:
    python extract_frames.py --input ./videos --output ./frames \\
        --route ./route_changes.csv --fps 1 --width 1280

    # MVP mode: skip videos entirely in daylight
    python extract_frames.py --input ./videos --output ./frames \\
        --route ./route_changes.csv --skip-daylight-videos

Dependencies:
    pip install astral pytz
    ffmpeg must be installed on the system
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytz
from astral import LocationInfo
from astral.sun import sun

TZ = pytz.timezone("Europe/Istanbul")
DEFAULT_VIDEO_DURATION_SEC = 60  # used for skip-daylight check; safe upper bound

# Filename pattern: matches a 14-digit YYYYMMDDHHMMSS sequence.
# Ignores prefixes (h_, t_) and suffixes (_0060).
TIMESTAMP_PATTERN = re.compile(r"(\d{14})")


def parse_video_start_time(filename: str):
    """Extract the recording start time from the filename. Returns None if unparseable."""
    m = TIMESTAMP_PATTERN.search(filename)
    if not m:
        return None
    try:
        naive_dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        return TZ.localize(naive_dt)
    except ValueError:
        return None


def light_condition(dt: datetime, city: str, lat: float, lon: float) -> str:
    """Classify a given time + location as daylight / twilight / night."""
    loc = LocationInfo(city, "TR", "Europe/Istanbul", lat, lon)
    s = sun(loc.observer, date=dt.date(), tzinfo=TZ)
    # astral order: dawn < sunrise < noon < sunset < dusk
    if dt < s["dawn"]:
        return "night"
    elif dt < s["sunrise"]:
        return "twilight"
    elif dt < s["sunset"]:
        return "daylight"
    elif dt < s["dusk"]:
        return "twilight"
    else:
        return "night"


def load_route_changes(csv_path: Path) -> list:
    """Load route transition points from CSV, sorted ascending by start_time.

    Returns list of (start_time, city, lat, lon) tuples.
    Expected columns: start_time, city, lat, lon
    """
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if len(row) < 4:
                continue
            # Skip header row if present
            if not header_skipped and row[0].strip().lower() in (
                "start_time",
                "starttime",
                "time",
            ):
                header_skipped = True
                continue
            header_skipped = True
            time_str = row[0].strip()
            city = row[1].strip()
            lat_str = row[2].strip()
            lon_str = row[3].strip()
            if not time_str or time_str.startswith("#"):
                continue
            try:
                dt = datetime.fromisoformat(time_str)
                if dt.tzinfo is None:
                    dt = TZ.localize(dt)
                lat = float(lat_str)
                lon = float(lon_str)
                rows.append((dt, city, lat, lon))
            except ValueError as e:
                print(
                    f"WARNING: cannot parse row '{row}' in route file: {e}",
                    file=sys.stderr,
                )
                continue

    rows.sort(key=lambda x: x[0])
    return rows


def get_location_for_time(dt: datetime, route: list):
    """Find the (city, lat, lon) for a given timestamp using route transition points.

    Returns (city, lat, lon) or None if no match.
    """
    if not route:
        return None
    matching = [(t, c, lat, lon) for t, c, lat, lon in route if t <= dt]
    if not matching:
        # No matching entry — use the earliest entry as fallback
        # (route is sorted, so route[0] is earliest)
        _, c, lat, lon = route[0]
        return (c, lat, lon)
    _, c, lat, lon = matching[-1]
    return (c, lat, lon)


def is_video_fully_daylight(
    start_dt: datetime,
    city: str,
    lat: float,
    lon: float,
    video_duration_sec: int = DEFAULT_VIDEO_DURATION_SEC,
) -> bool:
    """Check if a video falls entirely within the daylight window for its location.

    Returns True only if BOTH start and end of the video are 'daylight'.
    """
    end_dt = start_dt + timedelta(seconds=video_duration_sec)
    return (
        light_condition(start_dt, city, lat, lon) == "daylight"
        and light_condition(end_dt, city, lat, lon) == "daylight"
    )


def extract_frames_for_video(
    video_path: Path, output_dir: Path, fps: int, width: int
) -> int:
    """Extract frames from a single video with ffmpeg. Returns count of frames written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video_path.stem}_%06d.jpg"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-2",
        "-q:v",
        "3",
        "-an",  # no audio
        str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {video_path.name}\n  stderr: {result.stderr}", file=sys.stderr)
        return 0

    return len(list(output_dir.glob("*.jpg")))


def index_frames(
    video_path: Path, output_dir: Path, fps: int, city: str, lat: float, lon: float
) -> list:
    """Build CSV rows for the extracted frames of a video."""
    start_dt = parse_video_start_time(video_path.name)
    if start_dt is None:
        return []

    rows = []
    for frame_path in sorted(output_dir.glob("*.jpg")):
        try:
            frame_idx = int(frame_path.stem.split("_")[-1])
        except ValueError:
            continue

        seconds_into_video = (frame_idx - 1) / fps
        frame_dt = start_dt + timedelta(seconds=seconds_into_video)

        rows.append(
            {
                "frame_path": str(frame_path.resolve()),
                "video": video_path.name,
                "city": city,
                "frame_idx": frame_idx,
                "timestamp": frame_dt.isoformat(),
                "date": frame_dt.date().isoformat(),
                "time": frame_dt.time().isoformat(timespec="seconds"),
                "light": light_condition(frame_dt, city, lat, lon),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=Path("."), help="Video folder")
    parser.add_argument(
        "--output", type=Path, default=Path("./frames"), help="Output folder"
    )
    parser.add_argument(
        "--fps", type=int, default=1, help="Frames per second to extract"
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width (px)")
    parser.add_argument(
        "--csv", type=Path, default=Path("./frames_index.csv"), help="Index CSV path"
    )
    parser.add_argument(
        "--route",
        type=Path,
        default=Path("./route_changes.csv"),
        help="CSV with start_time,city,lat,lon transition points",
    )
    parser.add_argument(
        "--skip-daylight-videos",
        action="store_true",
        help="Skip videos that are entirely in the daylight window "
        "(no frames extracted, ffmpeg not invoked). MVP mode.",
    )
    parser.add_argument("--ext", default="mp4", help="Video extension (default: mp4)")
    args = parser.parse_args()

    videos = sorted(args.input.glob(f"*.{args.ext}"))
    if not videos:
        print(f"Error: no .{args.ext} files in {args.input}.", file=sys.stderr)
        sys.exit(1)

    route = load_route_changes(args.route)
    if not route:
        print(f"ERROR: route file {args.route} is empty or unreadable", file=sys.stderr)
        sys.exit(1)
    print(f"{len(route)} route transition points loaded")
    print(
        f"  Earliest entry (used as default fallback): "
        f"{route[0][1]} ({route[0][2]}, {route[0][3]})"
    )

    print(f"{len(videos)} videos found. Target: {args.fps} fps, width {args.width}px")
    if args.skip_daylight_videos:
        print("MVP mode: videos entirely in daylight will be skipped")
    print()

    all_rows = []
    skipped_daylight = 0
    skipped_unparseable = 0

    for i, video in enumerate(videos, 1):
        start_dt = parse_video_start_time(video.name)
        if start_dt is None:
            print(
                f"[{i}/{len(videos)}] {video.name} -> timestamp not parseable, skipping"
            )
            skipped_unparseable += 1
            continue

        location = get_location_for_time(start_dt, route)
        city, lat, lon = location

        # MVP filter: skip if entirely daylight
        if args.skip_daylight_videos and is_video_fully_daylight(
            start_dt, city, lat, lon
        ):
            skipped_daylight += 1
            continue

        print(f"[{i}/{len(videos)}] {video.name} ({city})")
        out_dir = args.output / video.stem
        n_frames = extract_frames_for_video(video, out_dir, args.fps, args.width)
        rows = index_frames(video, out_dir, args.fps, city, lat, lon)
        print(f"  -> {n_frames} frames extracted, {len(rows)} rows indexed")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo frames indexed.")
        if skipped_daylight > 0:
            print(f"({skipped_daylight} videos skipped due to --skip-daylight-videos)")
        sys.exit(1)

    # Write CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print()
    print(f"Total: {len(all_rows)} frames -> {args.csv}")
    if skipped_daylight > 0:
        print(f"Skipped (daylight only): {skipped_daylight} videos")
    if skipped_unparseable > 0:
        print(f"Skipped (unparseable filename): {skipped_unparseable} videos")

    light_counts = {}
    for r in all_rows:
        light_counts[r["light"]] = light_counts.get(r["light"], 0) + 1
    print("Light condition distribution:")
    for k, v in sorted(light_counts.items()):
        print(f"  {k:10s}: {v}")


if __name__ == "__main__":
    main()
