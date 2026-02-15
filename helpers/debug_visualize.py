"""Debug visualization utilities for the SAI3D segmenter pipeline.

Gated by the SAI3D_DEBUG environment variable. When SAI3D_DEBUG=1, each
checkpoint saves artifacts (images, PLY meshes, numpy arrays) to a debug
output directory. When disabled, all methods are no-ops with zero overhead.

Usage:
    export SAI3D_DEBUG=1
    python segment.py <observations_path> <scene_path>

Debug output will be written to <work_dir>/debug/.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("sai3d-segmenter")


def is_debug_enabled() -> bool:
    return os.environ.get("SAI3D_DEBUG", "0") == "1"


def _random_colors_for_labels(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Assign a deterministic random RGB color to each unique label.

    Label 0 is treated as background and gets light gray.
    Returns (N, 3) float64 array in [0, 1].
    """
    rng = np.random.RandomState(seed)
    unique = np.unique(labels)
    color_map = {}
    for u in unique:
        if u <= 0:
            color_map[u] = np.array([0.9, 0.9, 0.9])
        else:
            color_map[u] = rng.rand(3)
    colors = np.array([color_map[int(l)] for l in labels])
    return colors


class DebugVisualizer:
    """Saves debug artifacts for each pipeline stage when SAI3D_DEBUG=1."""

    def __init__(self, output_dir: Path):
        self.enabled = is_debug_enabled()
        self.output_dir = output_dir
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug visualizer ENABLED -- output dir: %s", self.output_dir)
        else:
            logger.debug("Debug visualizer disabled (set SAI3D_DEBUG=1 to enable)")

    # ------------------------------------------------------------------
    # Checkpoint 1: Input frames
    # ------------------------------------------------------------------
    def save_frames(self, frames, max_frames: int = 5) -> None:
        """Save a few RGB + depth frames as PNGs."""
        if not self.enabled:
            return
        import matplotlib.pyplot as plt

        out = self.output_dir / "frames"
        out.mkdir(exist_ok=True)

        for i, frame in enumerate(frames[:max_frames]):
            # RGB
            rgb = frame.color.cpu().numpy()  # (H, W, 3) float [0,1]
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            plt.imsave(str(out / f"{frame.name}_rgb.png"), rgb_uint8)

            # Depth (colormapped)
            if frame.depth is not None:
                depth = frame.depth.cpu().numpy()
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                im = ax.imshow(depth, cmap="turbo")
                ax.set_title(f"Depth -- {frame.name}")
                fig.colorbar(im, ax=ax, label="meters")
                fig.savefig(str(out / f"{frame.name}_depth.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)

        logger.info("[DEBUG] Saved %d frame visualizations to %s", min(len(frames), max_frames), out)

    # ------------------------------------------------------------------
    # Checkpoint 2: TSDF mesh
    # ------------------------------------------------------------------
    def save_mesh(self, mesh, filename: str = "mesh_tsdf.ply") -> None:
        """Save the reconstructed mesh as a colored PLY."""
        if not self.enabled:
            return
        import open3d as o3d

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh)
        logger.info(
            "[DEBUG] Saved TSDF mesh: %d verts, %d tris -> %s",
            len(mesh.vertices),
            len(mesh.triangles),
            path,
        )

    # ------------------------------------------------------------------
    # Checkpoint 3: Superpoints
    # ------------------------------------------------------------------
    def save_superpoints(self, mesh, seg_ids: np.ndarray, filename: str = "mesh_superpoints.ply") -> None:
        """Save mesh colored by superpoint assignment."""
        if not self.enabled:
            return
        import copy

        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(seg_ids, seed=123)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        unique_sp = np.unique(seg_ids)
        sizes = np.array([np.sum(seg_ids == s) for s in unique_sp])
        logger.info(
            "[DEBUG] Superpoints: %d unique, size min=%d median=%d max=%d -> %s",
            len(unique_sp),
            sizes.min(),
            int(np.median(sizes)),
            sizes.max(),
            path,
        )

    # ------------------------------------------------------------------
    # Checkpoint 4: 2D Semantic-SAM masks
    # ------------------------------------------------------------------
    def save_masks_2d(self, frames, masks_np: np.ndarray, max_frames: int = 8) -> None:
        """Save colorized 2D mask PNGs for the first N frames."""
        if not self.enabled:
            return
        import matplotlib.pyplot as plt

        out = self.output_dir / "masks"
        out.mkdir(exist_ok=True)

        for i in range(min(masks_np.shape[0], max_frames)):
            mask = masks_np[i]  # (H, W) int
            frame = frames[i]

            # Colorized mask
            colors = _random_colors_for_labels(mask.ravel(), seed=42 + i).reshape(mask.shape + (3,))
            colors_uint8 = (colors * 255).astype(np.uint8)

            # Side by side: RGB | mask
            rgb = frame.color.cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].imshow(rgb_uint8)
            axes[0].set_title(f"RGB -- {frame.name}")
            axes[0].axis("off")
            axes[1].imshow(colors_uint8)
            axes[1].set_title(f"SAM Masks -- {int(mask.max())} instances")
            axes[1].axis("off")
            fig.savefig(str(out / f"{frame.name}_masks.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

        total_masks = sum(int(masks_np[i].max()) for i in range(masks_np.shape[0]))
        logger.info(
            "[DEBUG] Saved 2D mask visualizations (%d frames, ~%d total mask instances) -> %s",
            min(masks_np.shape[0], max_frames),
            total_masks,
            out,
        )

    # ------------------------------------------------------------------
    # Checkpoint 5: Workspace voxels
    # ------------------------------------------------------------------
    def save_workspace_voxels(self, workspace_voxels, filename: str = "workspace_voxels.ply") -> None:
        """Save workspace voxel centers as a point cloud PLY."""
        if not self.enabled:
            return
        import open3d as o3d

        voxels = workspace_voxels.get_voxels()
        origin = workspace_voxels.origin
        vs = workspace_voxels.voxel_size
        centers = np.array([origin + np.asarray(v.grid_index) * vs + vs / 2 for v in voxels])

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
        pcd.paint_uniform_color([0.2, 0.7, 0.3])

        path = self.output_dir / filename
        o3d.io.write_point_cloud(str(path), pcd)
        logger.info("[DEBUG] Saved workspace voxels (%d voxels) -> %s", len(voxels), path)

    # ------------------------------------------------------------------
    # Checkpoint 6: SAI3D segmented mesh (after region growing)
    # ------------------------------------------------------------------
    def save_segmented_mesh(
        self,
        mesh,
        vertex_labels: np.ndarray,
        filename: str = "mesh_segmented.ply",
    ) -> None:
        """Save mesh colored by SAI3D instance labels (before filtering)."""
        if not self.enabled:
            return
        import copy

        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(vertex_labels, seed=77)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        n_labels = len(np.unique(vertex_labels[vertex_labels > 0]))
        logger.info("[DEBUG] Segmented mesh: %d instances -> %s", n_labels, path)

    # ------------------------------------------------------------------
    # Checkpoint 7: Filtered mesh (workspace + table removal)
    # ------------------------------------------------------------------
    def save_filtered_mesh(
        self,
        mesh,
        vertex_labels_before: np.ndarray,
        vertex_labels_after: np.ndarray,
        table_id: int,
        valid_ids: np.ndarray,
        filename: str = "mesh_filtered.ply",
    ) -> None:
        """Save mesh after workspace filtering and table removal."""
        if not self.enabled:
            return
        import copy

        import open3d as o3d

        mesh_copy = copy.deepcopy(mesh)
        colors = _random_colors_for_labels(vertex_labels_after, seed=77)
        mesh_copy.vertex_colors = o3d.utility.Vector3dVector(colors)

        path = self.output_dir / filename
        o3d.io.write_triangle_mesh(str(path), mesh_copy)

        before_ids = set(np.unique(vertex_labels_before)) - {0}
        after_ids = set(np.unique(vertex_labels_after)) - {0}
        removed = before_ids - after_ids
        logger.info(
            "[DEBUG] Filtering: %d -> %d instances (removed %s, table_id=%d) -> %s",
            len(before_ids),
            len(after_ids),
            removed if removed else "none",
            table_id,
            path,
        )

    # ------------------------------------------------------------------
    # Checkpoint 8: Back-projected pixel masks
    # ------------------------------------------------------------------
    def save_pixel_masks(
        self,
        frames,
        instance_groups: Dict[str, np.ndarray],
        max_frames: int = 8,
    ) -> None:
        """Save colorized per-frame instance mask PNGs (back-projected from mesh)."""
        if not self.enabled:
            return
        import matplotlib.pyplot as plt

        out = self.output_dir / "pixel_masks"
        out.mkdir(exist_ok=True)

        saved = 0
        for frame in frames:
            if saved >= max_frames:
                break
            mask = instance_groups.get(frame.name)
            if mask is None:
                continue

            colors = _random_colors_for_labels(mask.ravel(), seed=77).reshape(mask.shape + (3,))
            colors_uint8 = (colors * 255).astype(np.uint8)

            rgb = frame.color.cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes[0].imshow(rgb_uint8)
            axes[0].set_title(f"RGB -- {frame.name}")
            axes[0].axis("off")
            axes[1].imshow(colors_uint8)
            n_inst = len(np.unique(mask[mask > 0]))
            axes[1].set_title(f"Pixel Masks -- {n_inst} instances")
            axes[1].axis("off")
            fig.savefig(str(out / f"{frame.name}_pixel_masks.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved += 1

        logger.info("[DEBUG] Saved %d back-projected pixel mask visualizations -> %s", saved, out)
