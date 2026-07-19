#!/usr/bin/env python3
"""
update_gallery.py — sync the group photo gallery.

Scans a source folder of raw photos (default: ~/Desktop/group_photos) for any
images that are not yet published on the website, converts/resizes them to
web-friendly sizes, and writes them into imgs/gallery/ (full) and
imgs/gallery/thumbs/ (grid thumbnails). It also (re)builds _data/gallery.yml,
the manifest the Jekyll gallery page reads, sorted by date the photo was taken.

Re-running is safe and incremental: photos already published are skipped, so
you can just drop new photos into the source folder and run this again.

No third-party dependencies — uses the macOS built-ins `sips` (convert/resize)
and `mdls` (read the capture date). Run from anywhere:

    python3 update_gallery.py
    python3 update_gallery.py --src /path/to/photos   # override source folder
    python3 update_gallery.py --force                 # reprocess everything
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# Resolve paths relative to this script so it works from any cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.expanduser("~/Desktop/group_photos")
GALLERY_DIR = os.path.join(ROOT, "imgs", "gallery")
THUMBS_DIR = os.path.join(GALLERY_DIR, "thumbs")
DATA_FILE = os.path.join(ROOT, "_data", "gallery.yml")

# Longest-edge pixel sizes / JPEG quality for the two generated variants.
FULL_MAX = 1600
FULL_Q = 82
THUMB_MAX = 600
THUMB_Q = 70

SRC_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}


def log(msg):
    print(msg, flush=True)


def sanitize(basename):
    """Turn an arbitrary source filename into a stable, url-safe .jpg name."""
    stem = os.path.splitext(basename)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._") or "photo"
    return stem + ".jpg"


def date_from_filename(name):
    """Best-effort capture date parsed from common camera/export filenames.

    Handles patterns like PXL_20240311_..., 20230508_185417, IMG-20230510-WA...,
    img20230510115115, and 13-digit epoch-millisecond names (WhatsApp, etc.).
    Returns an ISO datetime string or None.
    """
    # Contiguous YYYYMMDD, optionally followed by an HHMMSS run of digits.
    for m in re.finditer(r"(20[0-3]\d)(\d{2})(\d{2})(\d{6}|\d{4})?", name):
        y, mo, d, tail = m.group(1), m.group(2), m.group(3), m.group(4)
        try:
            hh, mm, ss = "12", "00", "00"
            if tail and len(tail) == 6:
                hh, mm, ss = tail[0:2], tail[2:4], tail[4:6]
            dt = datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss))
            if 2005 <= dt.year <= 2035:
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    # Separated YYYY-MM-DD / YYYY_MM_DD / YYYY.MM.DD (e.g. WhatsApp exports).
    for m in re.finditer(r"(20[0-3]\d)[-_.](0[1-9]|1[0-2])[-_.](0[1-9]|[12]\d|3[01])", name):
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 12, 0, 0)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    # 13-digit epoch milliseconds anywhere in the name (e.g. 1676643947800.jpeg,
    # FB_IMG_1624664474009.jpg).
    for m in re.finditer(r"(?<!\d)(\d{13})(?!\d)", os.path.splitext(name)[0]):
        try:
            dt = datetime.fromtimestamp(int(m.group(1)) / 1000)
            if 2005 <= dt.year <= 2035:
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except (ValueError, OverflowError, OSError):
            continue
    return None


def _mdls_datetime(path, attr):
    """Read a datetime metadata attribute via mdls; return aware datetime or None."""
    try:
        out = subprocess.run(
            ["mdls", "-name", attr, "-raw", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out and out != "(null)":
            return datetime.strptime(out, "%Y-%m-%d %H:%M:%S %z")
    except Exception:
        pass
    return None


def capture_date(path):
    """Return an ISO datetime string for when the photo was taken, or "" if unknown.

    Priority:
      1. A date embedded in the filename (reliable for exported/renamed photos
         whose EXIF was stripped, e.g. PXL_/IMG-YYYYMMDD/epoch names).
      2. The embedded content-creation (EXIF) date — but ONLY if it predates the
         moment the file was written to this disk. If the two are essentially
         equal, the file has no real EXIF date (it's just the import time), so we
         treat the photo as undated rather than stamping it "today".

    Undated photos return "" and are sorted last by the gallery.
    """
    name_date = date_from_filename(os.path.basename(path))
    if name_date:
        return name_date

    content = _mdls_datetime(path, "kMDItemContentCreationDate")
    if content:
        fs_created = _mdls_datetime(path, "kMDItemFSCreationDate")
        # Real EXIF date is meaningfully older than when the file landed here.
        if fs_created is None or content < fs_created - timedelta(hours=12):
            return content.astimezone().strftime("%Y-%m-%dT%H:%M:%S")

    return ""  # no trustworthy capture date


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_month_year(text):
    """Parse loose month/year input into (month, year), or None.

    Accepts e.g. "03/2024", "2024-03", "March 2024", "Mar 2024", "2024"
    (year only -> mid-year). Returns None if unparseable.
    """
    text = text.strip().lower()
    if not text:
        return None
    year = month = None
    # Any 4-digit year in a sane range.
    ym = re.search(r"(20[0-3]\d|19\d\d)", text)
    if ym:
        year = int(ym.group(1))
    # A month name.
    for name, num in MONTHS.items():
        if name in text:
            month = num
            break
    # A numeric month (1-12) that isn't the year.
    if month is None:
        for tok in re.findall(r"\d{1,2}", text):
            v = int(tok)
            if 1 <= v <= 12:
                month = v
                break
    if year is None:
        return None
    return (month or 6, year)  # default to mid-year if only a year was given


def prompt_for_date(name, src_path):
    """Ask the user for the month/year of an undated photo. Returns "" if skipped."""
    if not sys.stdin.isatty():
        return ""
    log("")
    log(f"  ? No date found for: {name}")
    try:
        subprocess.run(["open", src_path], check=False)  # show it in Preview
        log("    (opened in Preview so you can see it)")
    except Exception:
        pass
    while True:
        try:
            ans = input("    Month & year taken (e.g. 03/2024, 'Mar 2024', or Enter to skip): ")
        except EOFError:
            return ""
        if not ans.strip():
            log("    -> skipped (photo will sort last as undated)")
            return ""
        parsed = parse_month_year(ans)
        if parsed:
            month, year = parsed
            iso = datetime(year, month, 1, 12, 0, 0).strftime("%Y-%m-%dT%H:%M:%S")
            log(f"    -> set to {iso[:7]}")
            return iso
        log("    Couldn't read that — try like '03/2024' or 'March 2024'.")


def sips_resize(src, dst, max_edge, quality):
    """Convert `src` to a JPEG at `dst`, longest edge <= max_edge."""
    subprocess.run(
        ["sips", "-s", "format", "jpeg", "-Z", str(max_edge),
         "--setProperty", "formatOptions", str(quality),
         src, "--out", dst],
        capture_output=True, text=True, check=True,
    )


def read_manifest():
    """Parse the existing gallery.yml into {file: {date, caption}} (minimal parser)."""
    entries = {}
    if not os.path.exists(DATA_FILE):
        return entries
    cur = None
    with open(DATA_FILE) as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"- file:\s*(.+)$", line.strip())
            if m:
                cur = m.group(1).strip().strip('"')
                entries[cur] = {}
                continue
            if cur:
                m = re.match(r"(\w+):\s*(.*)$", line.strip())
                if m:
                    entries[cur][m.group(1)] = m.group(2).strip().strip('"')
    return entries


def write_manifest(items):
    """Write the manifest sorted newest-first."""
    items = sorted(items, key=lambda e: e.get("date", ""), reverse=True)
    lines = [
        "# Auto-generated by update_gallery.py — do not edit by hand.",
        "# Each entry: full image in /imgs/gallery/, thumb in /imgs/gallery/thumbs/.",
        "",
    ]
    for e in items:
        lines.append(f'- file: "{e["file"]}"')
        lines.append(f'  date: "{e["date"]}"')
        if e.get("caption"):
            lines.append(f'  caption: "{e["caption"]}"')
        if e.get("nodate"):
            lines.append("  nodate: true")
    with open(DATA_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    return items


def main():
    ap = argparse.ArgumentParser(description="Sync the group photo gallery.")
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help=f"Source photo folder (default: {DEFAULT_SRC})")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess photos even if already published.")
    ap.add_argument("--no-prompt", action="store_true",
                    help="Don't ask for a date on undated photos; leave them undated.")
    ap.add_argument("--update_deleted", action="store_true",
                    help="Also remove published photos whose source was deleted "
                         "from the source folder (off by default).")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        log(f"ERROR: source folder not found: {args.src}")
        return 1

    os.makedirs(THUMBS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

    manifest = read_manifest()

    # Discover source photos.
    sources = sorted(
        f for f in os.listdir(args.src)
        if not f.startswith(".") and os.path.splitext(f)[1].lower() in SRC_EXTS
    )

    prompting = (not args.no_prompt) and sys.stdin.isatty()

    added, dated, skipped, failed = 0, 0, 0, 0
    for name in sources:
        src_path = os.path.join(args.src, name)
        out_name = sanitize(name)
        full_path = os.path.join(GALLERY_DIR, out_name)
        thumb_path = os.path.join(THUMBS_DIR, out_name)
        meta = manifest.get(out_name)

        already = os.path.exists(full_path) and os.path.exists(thumb_path)

        # Already published: don't re-resize. Just resolve a missing date if we can.
        if already and not args.force:
            if meta and meta.get("date"):
                skipped += 1
                continue  # already has a good date
            # No stored date — try to derive one cheaply (metadata / filename) now.
            date = capture_date(src_path)
            if not date:
                if (meta and meta.get("nodate")) or not prompting:
                    skipped += 1
                    continue  # can't derive, and already skipped or non-interactive
                date = prompt_for_date(name, src_path)
            m = manifest.setdefault(out_name, {})
            if date:
                m["date"] = date
                m.pop("nodate", None)
                dated += 1
                log(f"  ~ dated {out_name}  ({date[:10]})")
            else:
                m["date"] = ""
                m["nodate"] = "true"  # asked & skipped — don't nag next run
                skipped += 1
            continue

        # New photo, or --force: (re)generate the resized variants.
        try:
            date = capture_date(src_path)
            if not date:
                prev = meta.get("date") if meta else None
                if prev:  # keep a previously-set (likely manual) date across --force
                    date = prev
                elif prompting:
                    date = prompt_for_date(name, src_path)
            sips_resize(src_path, full_path, FULL_MAX, FULL_Q)
            # Thumb is derived from the already-resized full jpg (faster).
            sips_resize(full_path, thumb_path, THUMB_MAX, THUMB_Q)
        except subprocess.CalledProcessError as e:
            failed += 1
            log(f"  ! failed: {name}\n    {e.stderr.strip()}")
            continue

        m = manifest.setdefault(out_name, {})
        m["date"] = date
        if not date and prompting:
            m["nodate"] = "true"  # asked & skipped
        else:
            m.pop("nodate", None)
        added += 1
        log(f"  + {name} -> {out_name}  ({date[:10] or 'undated'})")

    # Optionally prune photos whose source was removed (opt-in: --update_deleted).
    removed = 0
    expected = {sanitize(name) for name in sources}
    published = {f for f in os.listdir(GALLERY_DIR)
                 if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(GALLERY_DIR, f))}
    stale = sorted(published - expected)
    if stale and not args.update_deleted:
        log(f"  ({len(stale)} published photo(s) no longer in the source folder; "
            f"pass --update_deleted to remove them)")
    elif stale and not sources:
        # Safety: an empty source folder is almost always a mistake, not an
        # instruction to wipe the whole gallery. Refuse and let the user decide.
        log(f"  ! source folder has no photos — refusing to prune {len(stale)} "
            f"published photo(s). Delete imgs/gallery/ manually if you really "
            f"mean to empty the gallery.")
    elif stale:
        for out_name in stale:
            for path in (os.path.join(GALLERY_DIR, out_name),
                         os.path.join(THUMBS_DIR, out_name)):
                if os.path.exists(path):
                    os.remove(path)
            manifest.pop(out_name, None)
            removed += 1
            log(f"  - removed {out_name} (source deleted)")

    # Rebuild the manifest from what actually exists on disk.
    items = []
    for out_name, meta in manifest.items():
        full_path = os.path.join(GALLERY_DIR, out_name)
        if not os.path.exists(full_path):
            continue  # source removed / renamed — drop stale entry
        date = meta.get("date")
        if date is None:  # never computed (absent key) — try once; "" is a kept value
            date = capture_date(full_path)
        item = {"file": out_name, "date": date}
        if meta.get("caption"):
            item["caption"] = meta["caption"]
        if meta.get("nodate"):
            item["nodate"] = "true"
        items.append(item)

    items = write_manifest(items)

    log("")
    log(f"Done. {added} added, {dated} newly dated, {removed} removed, "
        f"{skipped} unchanged, {failed} failed.")
    log(f"Gallery now has {len(items)} photos -> {os.path.relpath(DATA_FILE, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
