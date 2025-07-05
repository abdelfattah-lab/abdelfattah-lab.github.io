#!/usr/bin/env python3
"""
extract_collabs_keep_spacing.py  ─  Convert “- name:” under ‘authors’
to “- key:” while
  • preserving all original whitespace
  • touching nothing outside the ‘authors’ list
  • giving every unique full name a single, stable key

Run:
    python extract_collabs_keep_spacing.py publications/  > collaborators.yml
"""

from __future__ import annotations
import argparse
import pathlib
import re
import sys
import yaml

# ───────── regexes ──────────────────────────────────────────
AUTHORS_LINE = re.compile(r'^(\s*)authors:\s*$')
FIELD_START  = re.compile(r'^(\s*)(\w+):')
NAME_ITEM    = re.compile(r'^(\s*)- name:\s*(.+?)\s*$')
KEY_ITEM     = re.compile(r'^(\s*)- key:\s*(\w+)\s*$')

# ───────── helpers ─────────────────────────────────────────
def fresh_key(first_name: str, taken: set[str]) -> str:
    """Return a unique lowercase key based on *first_name*."""
    base = re.sub(r'[^A-Za-z]', '', first_name).lower()
    key, n = base, 1
    while key in taken:
        key = f'{base}{n}'
        n += 1
    taken.add(key)
    return key


def scan_front_matter(lines: list[str]) -> tuple[int, int] | None:
    if not lines or lines[0].strip() != '---':
        return None
    try:
        end = lines.index('---\n', 1)
    except ValueError:
        sys.exit("Unterminated front-matter fence")
    return 0, end


def safe_read(path: pathlib.Path):
    try:
        return path.read_text(encoding='utf-8').splitlines(keepends=True)
    except UnicodeDecodeError:
        return None                                    # skip binary / non-utf-8


def process(path: pathlib.Path,
            used_keys: set[str],
            name2key: dict[str, str],
            collab: dict[str, dict]):

    text = safe_read(path)
    if text is None:
        return

    span = scan_front_matter(text)
    if span is None:
        return

    fm_start, fm_end = span
    changed = False
    i = fm_start + 1

    while i < fm_end:
        m_auth = AUTHORS_LINE.match(text[i])
        if not m_auth:
            i += 1
            continue

        base_indent = len(m_auth.group(1))
        i += 1

        while i < fm_end:
            line = text[i]
            if FIELD_START.match(line) and \
               len(FIELD_START.match(line).group(1)) <= base_indent:
                break        # end of authors list

            # Existing key: record but no mapping (we don't know the full name)
            m_key = KEY_ITEM.match(line)
            if m_key:
                used_keys.add(m_key.group(2))

            # “- name:” → convert / look-up
            m_name = NAME_ITEM.match(line)
            if m_name:
                indent, full = m_name.groups()

                # ‣ Re-use existing key if we’ve seen the name before
                if full in name2key:
                    key = name2key[full]
                else:
                    key = fresh_key(full.split()[0], used_keys)
                    name2key[full] = key
                    collab.setdefault(
                        key,
                        {"name": full, "collab": True, "affiliation": ""}
                    )

                text[i] = f'{indent}- key: {key}\n'
                changed = True
            i += 1

    if changed:
        path.write_text(''.join(text), encoding='utf-8')


def main(root: pathlib.Path):
    used_keys: set[str] = set()
    collaborators: dict[str, dict] = {}
    name_to_key: dict[str, str] = {}

    # Only process publication source files (adjust extension if needed)
    for p in root.rglob('*.md'):
        if p.is_file():
            process(p, used_keys, name_to_key, collaborators)

    yaml.dump(collaborators, sys.stdout, sort_keys=False, allow_unicode=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Normalise author lists without altering other ‘name’ fields."
    )
    ap.add_argument('directory', help='Directory containing publication files')
    main(pathlib.Path(ap.parse_args().directory))
