#!/usr/bin/env python3
"""Simple PLY/PCD viewer using Open3D.

Usage:
    python view_ply.py <file.ply> [file2.ply ...]
    python view_ply.py debug/                       # view all PLY/PCD files in a directory

Controls (Open3D viewer):
    Mouse left   - Rotate
    Mouse wheel  - Zoom
    Mouse right  - Pan
    R            - Reset view
    Q / Esc      - Close window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import open3d as o3d


def load_geometry(path: Path):
    """Load a PLY or PCD file as either a mesh or point cloud."""
    suffix = path.suffix.lower()
    if suffix == ".ply":
        # Try mesh first (has triangles), fall back to point cloud
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            return mesh
        # No triangles — treat as point cloud
        pcd = o3d.io.read_point_cloud(str(path))
        return pcd
    elif suffix == ".pcd":
        return o3d.io.read_point_cloud(str(path))
    else:
        print(f"Unsupported file format: {suffix}", file=sys.stderr)
        return None


def collect_files(paths: list[str]) -> list[Path]:
    """Expand directories into their contained PLY/PCD files."""
    result = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in ("*.ply", "*.pcd"):
                result.extend(sorted(path.rglob(ext)))
        elif path.is_file():
            result.append(path)
        else:
            print(f"Warning: {p} does not exist, skipping", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="View PLY/PCD files using Open3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="PLY/PCD files or directories to view",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all files in a single window (default: one window per file)",
    )
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("No PLY/PCD files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    if args.all:
        # Load everything into one window
        geometries = []
        for f in files:
            geom = load_geometry(f)
            if geom is not None:
                geometries.append(geom)
        if geometries:
            o3d.visualization.draw_geometries(
                geometries,
                window_name="SAI3D Debug Viewer",
                width=1280,
                height=720,
            )
    else:
        # One window per file, sequential
        for f in files:
            geom = load_geometry(f)
            if geom is None:
                continue
            o3d.visualization.draw_geometries(
                [geom],
                window_name=f"SAI3D: {f.name}",
                width=1280,
                height=720,
            )


if __name__ == "__main__":
    main()
