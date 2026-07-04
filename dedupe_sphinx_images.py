#!/usr/bin/env python3
"""
dedupe_sphinx_images.py

Fixes Sphinx's image-flattening collision problem by giving every image a
project-unique filename, based on the name of its parent folder.

Example:
    project/.../basket/images/front_view.JPG
        ->  project/.../basket/images/basket_front_view.JPG

    project/.../chesterfield/images/front_view.JPG
        ->  project/.../chesterfield/images/chesterfield_front_view.JPG

After renaming, it rewrites every reference to the old filename inside
Markdown/reST/text files so nothing breaks (MyST ![]() syntax, {image} /
{figure} directives, sphinx-design grid-item-card images, raw HTML <img>,
and reST image:: / figure:: directives all use a plain filename or relative
path, so a text substitution covers them).

USAGE
-----
    # 1. Dry run first (default) -- shows what WOULD happen, changes nothing
    python dedupe_sphinx_images.py /path/to/source

    # 2. Once the plan looks right, actually apply it
    python dedupe_sphinx_images.py /path/to/source --apply

    # Optional: also write a text report of every rename + replacement
    python dedupe_sphinx_images.py /path/to/source --apply --report report.txt

NOTES
-----
- Only files inside directories literally named "images" are renamed.
- The "project name" used as the prefix is the name of the parent folder
  of that images/ directory (e.g. "basket", "solid_signpost", "table copy 1").
- Folder names with spaces/uppercase are slugified to snake_case for the
  filename prefix (e.g. "table copy 1" -> "table_copy_1"), but the folder
  itself is left untouched -- only files inside images/ are renamed.
- Already-prefixed files (e.g. re-running the script) are skipped, so it's
  safe to run more than once.
- Text substitution is restricted to files that live inside the SAME
  project folder (or its parent, one level up, to catch an index page that
  links down into it) -- this avoids accidentally rewriting an identical
  filename that belongs to a *different* project elsewhere in the tree.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

TEXT_EXTENSIONS = {".md", ".rst", ".txt"}


def slugify(name: str) -> str:
    """Turn a folder name into a safe, readable filename prefix."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "project"


def find_image_dirs(root: Path):
    """Yield every directory literally named 'images' under root."""
    for p in root.rglob("images"):
        if p.is_dir():
            yield p


def find_text_files(*dirs):
    """Yield unique text files (.md/.rst/.txt) found under the given dirs."""
    seen = set()
    for d in dirs:
        if not d.exists():
            continue
        for ext in TEXT_EXTENSIONS:
            for f in d.rglob(f"*{ext}"):
                if f not in seen:
                    seen.add(f)
                    yield f


def plan_renames(root: Path):
    """
    Build a rename plan.

    Returns a list of dicts:
        {
            "project_dir": Path,
            "images_dir": Path,
            "renames": [(old_name, new_name), ...],
        }
    """
    plan = []
    for images_dir in find_image_dirs(root):
        project_dir = images_dir.parent
        prefix = slugify(project_dir.name)

        renames = []
        for f in sorted(images_dir.iterdir()):
            if not f.is_file():
                continue
            old_name = f.name
            if old_name.lower().startswith(prefix.lower() + "_"):
                continue  # already prefixed, skip (safe to re-run script)
            new_name = f"{prefix}_{old_name}"
            renames.append((old_name, new_name))

        if renames:
            plan.append(
                {
                    "project_dir": project_dir,
                    "images_dir": images_dir,
                    "renames": renames,
                }
            )
    return plan


def apply_renames(entry, dry_run: bool, log_lines: list):
    images_dir: Path = entry["images_dir"]
    for old_name, new_name in entry["renames"]:
        src = images_dir / old_name
        dst = images_dir / new_name
        line = f"RENAME  {src}  ->  {dst}"
        log_lines.append(line)
        print(line)
        if not dry_run:
            if dst.exists():
                print(f"  !! SKIPPED (target already exists): {dst}")
                continue
            shutil.move(str(src), str(dst))


def update_references(entry, root: Path, dry_run: bool, log_lines: list):
    """
    Rewrite references to old filenames inside nearby text files.

    Two separate zones, handled differently to avoid cross-project
    contamination (this matters here, since many projects reuse the exact
    same bare filenames like 'front_view.JPG'):

      1. INSIDE the project directory (recursively): bare-filename
         substitution is safe, because this subtree belongs to exactly one
         project.

      2. The project directory's PARENT, but *only files sitting directly
         in that parent* (never recursing into sibling project folders --
         that would risk rewriting an unrelated project's identical
         filename). For these outer files (e.g. an index/landing page),
         only path-qualified references such as
         "basket/images/front_view.JPG" are replaced -- never a bare
         filename -- since an index page may legitimately mention the same
         bare filename for a different project.
    """
    project_dir: Path = entry["project_dir"]
    parent_dir = project_dir.parent

    inner_files = list(find_text_files(project_dir))
    outer_files = [
        f for ext in TEXT_EXTENSIONS for f in parent_dir.glob(f"*{ext}")
    ]

    for old_name, new_name in entry["renames"]:
        bare_pattern = re.compile(r"(?<!\w)" + re.escape(old_name) + r"(?!\w)")

        # Zone 1: inside the project directory -- bare filename is safe
        for tf in inner_files:
            try:
                content = tf.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if old_name not in content:
                continue
            new_content, n = bare_pattern.subn(new_name, content)
            if n > 0:
                line = f"UPDATE  {tf}  ({old_name} -> {new_name}, {n} occurrence(s))"
                log_lines.append(line)
                print(line)
                if not dry_run:
                    tf.write_text(new_content, encoding="utf-8")

        # Zone 2: parent dir's own files only -- require the project-name
        # path segment so we never touch a different project's same-named file
        qualified_old = f"{project_dir.name}/images/{old_name}"
        qualified_new = f"{project_dir.name}/images/{new_name}"
        for tf in outer_files:
            try:
                content = tf.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if qualified_old not in content:
                continue
            n = content.count(qualified_old)
            new_content = content.replace(qualified_old, qualified_new)
            line = f"UPDATE  {tf}  ({qualified_old} -> {qualified_new}, {n} occurrence(s))"
            log_lines.append(line)
            print(line)
            if not dry_run:
                tf.write_text(new_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_dir", type=Path, help="Path to your Sphinx 'source' directory")
    parser.add_argument("--apply", action="store_true", help="Actually perform renames/edits (default is dry-run)")
    parser.add_argument("--report", type=Path, default=None, help="Optional path to write a text log of changes")
    args = parser.parse_args()

    root = args.source_dir.resolve()
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    dry_run = not args.apply
    mode = "DRY RUN (no changes will be made)" if dry_run else "APPLYING CHANGES"
    print(f"=== {mode} ===\nScanning: {root}\n")

    plan = plan_renames(root)

    if not plan:
        print("No colliding/unprefixed images found. Nothing to do.")
        return

    log_lines = []
    for entry in plan:
        print(f"\n--- Project: {entry['project_dir'].relative_to(root)} ---")
        apply_renames(entry, dry_run, log_lines)
        update_references(entry, root, dry_run, log_lines)

    print(f"\n=== Done. {sum(len(e['renames']) for e in plan)} file(s) affected across {len(plan)} project folder(s). ===")

    if dry_run:
        print("\nThis was a dry run -- no files were changed.")
        print("Re-run with --apply once this plan looks correct.")

    if args.report:
        args.report.write_text("\n".join(log_lines), encoding="utf-8")
        print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
