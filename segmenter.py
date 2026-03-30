"""SAI3D external segmenter bridge module.

Adapts the SAI3D pipeline (Semantic-SAM 2D masks + ScanNet Segmentator superpoints
+ progressive region growing) to the deg external-segmenter protocol so it can be
compared against frame-seg-init on the same TSDF mesh and workspace filter.
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import certifi
from plyfile import PlyData, PlyElement

# Prevent Open3D from loading its ML sub-module, which pulls in sklearn and
# triggers a numpy binary-incompatibility error in some environments.
import sys as _sys, types as _types

_sys.modules.setdefault("open3d.ml", _types.ModuleType("open3d.ml"))

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from initializerdefs import (
    InstanceMaskObjectsDef,
    ObjectSegmentations,
    Observations,
    ObservationFrame,
    SceneSetup,
)
from psdframe import Frame

from helpers.debug_visualize import DebugVisualizer

logger = logging.getLogger("sai3d-segmenter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEGMENTATOR_DIR = Path(__file__).parent / "Segmentator"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_URL = "https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth"
CHECKPOINT_FILENAME = "swinl_only_sam_many2many.pth"

# SAI3D default hyper-parameters
THRES_CONNECT_START = 0.9
THRES_CONNECT_END = 0.5
THRES_CONNECT_STAGES = 5
VIS_DIS = 0.15
MAX_NEIGHBOR_DISTANCE = 2
SIMILAR_METRIC = "2-norm"
THRES_MERGE = 200
VIEW_FREQ = 1  # use every frame (we already have a small set of frames)
DIS_DECAY = 0.5
THRES_TRUNC = 0.0
FROM_POINTS_THRES = 0
K_THRESH = 0.01
SEG_MIN_VERTS = 20

# ---------------------------------------------------------------------------
# helpers – ObservationFrame → Frame
# ---------------------------------------------------------------------------


def get_dataset_frame_from_observation_frame(observation_frame: ObservationFrame) -> Frame:
    """Copied from SegmentAnything3D/segmenter.py."""
    return Frame(
        id=observation_frame.id,
        name=observation_frame.name,
        color=torch.tensor(observation_frame.color).cuda(),
        X_WV=torch.tensor(observation_frame.X_WV),
        K=torch.tensor(observation_frame.K),
        depth=(torch.tensor(observation_frame.depth).cuda() if observation_frame.depth is not None else None),
    )


# ---------------------------------------------------------------------------
# TSDF mesh reconstruction (inline copy from frame-seg-init)
# ---------------------------------------------------------------------------
_x_rot = np.eye(4, dtype=np.float32)
_x_rot[:3, :3] = R.from_euler("x", 180, degrees=True).as_matrix()


def _to_cam_open3d(frame: Frame) -> o3d.camera.PinholeCameraParameters:
    """Convert a psdframe.Frame to Open3D camera parameters."""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(frame.w, frame.h, frame.fl_x, frame.fl_y, frame.cx, frame.cy)
    extrinsic = frame.X_VW_opencv.cpu().numpy()
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic
    return camera


def _post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int = 1000) -> o3d.geometry.TriangleMesh:
    """Remove small disconnected clusters from a mesh."""
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, _ = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    logger.info("mesh vertices raw %d → post %d", len(mesh.vertices), len(mesh_0.vertices))
    return mesh_0


@torch.no_grad()
def _extract_mesh_bounded(
    frames: List[Frame], voxel_size: float = 0.004, sdf_trunc: float = 0.02, depth_trunc: float = 3
) -> o3d.geometry.TriangleMesh:
    """TSDF integration and marching-cubes extraction."""
    logger.info("TSDF integration: voxel_size=%.4f  sdf_trunc=%.4f  depth_trunc=%.2f", voxel_size, sdf_trunc, depth_trunc)
    for frame in frames:
        assert frame.depth is not None and frame.color is not None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for frame in tqdm(frames, desc="TSDF integration"):
        rgb = frame.color.cpu().numpy()
        assert frame.depth is not None
        depth = frame.depth.cpu().numpy()

        ci = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        di = o3d.geometry.Image(depth)
        cam = _to_cam_open3d(frame)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(ci, di, depth_trunc=depth_trunc, convert_rgb_to_intensity=False, depth_scale=1.0)
        volume.integrate(rgbd, intrinsic=cam.intrinsic, extrinsic=cam.extrinsic)

    return volume.extract_triangle_mesh()


def _extract_mesh_bounded_with_res(frames: List[Frame], depth_trunc: float = 2, mesh_res: int = 1024) -> o3d.geometry.TriangleMesh:
    voxel_size = depth_trunc / mesh_res
    sdf_trunc = 5.0 * voxel_size
    raw_mesh = _extract_mesh_bounded(frames, voxel_size, sdf_trunc, depth_trunc)
    return _post_process_mesh(raw_mesh, cluster_to_keep=50)


# ---------------------------------------------------------------------------
# Workspace filtering (copied from SegmentAnything3D/segmenter.py)
# ---------------------------------------------------------------------------


def _erode_voxel_grid_xy(voxel_grid: o3d.geometry.VoxelGrid, layers: int) -> o3d.geometry.VoxelGrid:
    if layers <= 0 or not voxel_grid.has_voxels():
        return voxel_grid

    voxel_indices = [tuple(int(idx) for idx in voxel.grid_index) for voxel in voxel_grid.get_voxels()]
    xy_occupied = {(x, y) for x, y, _ in voxel_indices}

    for _ in range(layers):
        if not xy_occupied:
            break
        prev_xy = xy_occupied
        xy_occupied = {
            (x, y) for (x, y) in prev_xy if ((x - 1, y) in prev_xy and (x + 1, y) in prev_xy and (x, y - 1) in prev_xy and (x, y + 1) in prev_xy)
        }

    for voxel_index in voxel_indices:
        if (voxel_index[0], voxel_index[1]) not in xy_occupied:
            voxel_grid.remove_voxel(voxel_index)

    return voxel_grid


def get_workspace_voxels(scene: SceneSetup, shrink_xy_m: float = 0.1) -> o3d.geometry.VoxelGrid:
    table_xyz = scene.ground_gaussians.xyz
    table_plane = scene.ground_plane
    table_normal = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_pcd_extruded = np.array(table_xyz).copy()

    DESIRED_HEIGHT = 1.0
    VOXEL_SIZE = 0.05
    iters = int(DESIRED_HEIGHT / VOXEL_SIZE)
    for i in range(iters):
        new_points = table_xyz + table_normal * VOXEL_SIZE * i
        table_pcd_extruded = np.append(table_pcd_extruded, new_points, axis=0)
    for i in range(5):
        table_pcd_extruded = np.append(table_pcd_extruded, table_xyz - table_normal * VOXEL_SIZE * (i + 1), axis=0)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(table_pcd_extruded))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE * 2)

    layers = max(0, int(np.round(shrink_xy_m / voxel_grid.voxel_size)))
    return _erode_voxel_grid_xy(voxel_grid, layers)


# ---------------------------------------------------------------------------
# Table removal (copied from SegmentAnything3D/util.py)
# ---------------------------------------------------------------------------


def _get_instance_id_mask_for_frame(instance_id: int, masks: Dict[str, np.ndarray], frame: Frame) -> torch.Tensor:
    frame_mask = masks[frame.name]
    instance_mask = frame_mask == instance_id
    return torch.tensor(instance_mask, device=frame.color.device, dtype=torch.bool)


def determine_table_instance_id(
    frames: List[Frame],
    masks: Dict[str, np.ndarray],
    table_plane: Tuple[float, float, float, float],
    object_ids: np.ndarray,
) -> int:
    instance_ids = object_ids
    if len(instance_ids) == 0:
        return -1

    table_instance_candidates: List[int] = []
    table_instance_counts: List[int] = []

    for frame in frames:
        assert frame.depth is not None
        h, w = frame.depth.shape
        y, x = torch.meshgrid(
            torch.arange(h, device=frame.depth.device),
            torch.arange(w, device=frame.depth.device),
            indexing="ij",
        )
        valid_mask = frame.depth > 0
        Z = frame.depth
        X = (x - frame.cx) * Z / frame.fl_x
        Y = (y - frame.cy) * Z / frame.fl_y
        points = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=0)
        points = frame.X_WV_opencv.cuda() @ points.reshape(4, -1)
        points = points.reshape(4, h, w)

        a, b, c, d = table_plane
        plane_dist = (a * points[0] + b * points[1] + c * points[2] + d) / math.sqrt(a * a + b * b + c * c)
        table_mask = torch.abs(plane_dist) < 0.02
        table_mask = table_mask & valid_mask

        for instance_id in instance_ids:
            inst_id_val = int(instance_id.item()) if hasattr(instance_id, "item") else int(instance_id)
            instance_mask = _get_instance_id_mask_for_frame(inst_id_val, masks, frame)
            instance_mask_valid = instance_mask & valid_mask
            instance_mask_near_table = instance_mask & table_mask
            valid_count = instance_mask_valid.sum()
            if valid_count > 0 and instance_mask_near_table.sum() / valid_count > 0.7:
                table_instance_candidates.append(inst_id_val)
                table_instance_counts.append(instance_mask_near_table.sum().item())

    if len(table_instance_candidates) == 0:
        logger.warning("No table candidates found")
        return -1

    best_idx = int(np.argmax(table_instance_counts))
    return table_instance_candidates[best_idx]


# ---------------------------------------------------------------------------
# Semantic-SAM checkpoint + mask generation
# ---------------------------------------------------------------------------


def _ensure_checkpoint() -> Path:
    """Download Semantic-SAM SwinL checkpoint if not present."""
    ckpt_path = CHECKPOINT_DIR / CHECKPOINT_FILENAME
    if ckpt_path.exists():
        logger.info("Semantic-SAM checkpoint found at %s", ckpt_path)
        return ckpt_path
    logger.info("Downloading Semantic-SAM checkpoint from %s …", CHECKPOINT_URL)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    import ssl
    import certifi

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(CHECKPOINT_URL, context=ssl_context) as response:
        with open(ckpt_path, "wb") as out_file:
            out_file.write(response.read())
    logger.info("Checkpoint saved to %s", ckpt_path)
    return ckpt_path


def _build_mask_generator():
    """Build Semantic-SAM mask generator (instance level)."""
    ckpt_path = _ensure_checkpoint()
    from semantic_sam import SemanticSamAutomaticMaskGenerator, build_semantic_sam
    from semantic_sam.build_semantic_sam import __file__ as _semantic_sam_build_path
    from contextlib import contextmanager

    @contextmanager
    def _semantic_sam_repo_cwd():
        repo_root = Path(_semantic_sam_build_path).resolve().parent.parent
        current = Path.cwd()
        os.chdir(repo_root)
        try:
            yield
        finally:
            os.chdir(current)

    with _semantic_sam_repo_cwd():
        model = build_semantic_sam(model_type="L", ckpt=str(ckpt_path))
        generator = SemanticSamAutomaticMaskGenerator(model, level=[3])
        return generator


def _generate_masks_for_frame(frame: Frame, mask_generator, cache_dir: Optional[Path] = None) -> np.ndarray:
    """Generate per-pixel mask ids for a single frame using Semantic-SAM.

    Returns (H, W) int array with mask label per pixel (starting from 0, -1 = bg).
    """
    from helpers.sam_utils import get_sam_by_iou, my_prepare_image, num_to_natural

    # Prepare CHW tensor on CUDA from frame color
    color_np = frame.color.cpu().numpy()  # (H, W, 3) float32 0-1
    color_uint8 = (color_np * 255).astype(np.uint8)
    image_tensor = torch.from_numpy(color_uint8).permute(2, 0, 1).cuda()

    # Check cache
    if cache_dir is not None:
        cache_file = cache_dir / f"{frame.name}.npy"
        if cache_file.exists():
            return np.load(str(cache_file))

    group_ids = get_sam_by_iou(image_tensor, mask_generator)
    group_ids = num_to_natural(group_ids) + 1  # shift so 0 = background

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_dir / f"{frame.name}.npy"), group_ids)

    return group_ids


# ---------------------------------------------------------------------------
# ScanNet Segmentator binary
# ---------------------------------------------------------------------------


def _ensure_segmentator_binary() -> Path:
    """Build the Segmentator C++ binary if it doesn't exist."""
    binary = SEGMENTATOR_DIR / "segmentator"
    if binary.exists():
        return binary
    logger.info("Building Segmentator binary …")
    result = subprocess.run(["make", "-C", str(SEGMENTATOR_DIR)], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Segmentator build failed:\n%s", result.stderr)
        raise RuntimeError(f"Failed to build Segmentator: {result.stderr}")
    assert binary.exists(), f"Expected binary at {binary} after build"
    logger.info("Segmentator binary built at %s", binary)
    return binary


def _run_segmentator(ply_path: Path, k_thresh: float = K_THRESH, seg_min_verts: int = SEG_MIN_VERTS) -> Path:
    """Run Segmentator on a PLY mesh, returns path to the output segs.json."""
    binary = _ensure_segmentator_binary()
    ply_path = _ensure_segmentator_ply(ply_path)
    # Output path follows the convention: <basename>.<kThresh>.segs.json
    expected_output = ply_path.parent / f"{ply_path.stem}.{k_thresh}.segs.json"
    if expected_output.exists():
        logger.info("Superpoints already computed: %s", expected_output)
        return expected_output

    logger.info("Running Segmentator on %s (kThresh=%s, segMinVerts=%d) …", ply_path, k_thresh, seg_min_verts)
    result = subprocess.run(
        [str(binary), str(ply_path), str(k_thresh), str(seg_min_verts)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Segmentator failed:\n%s", result.stderr)
        raise RuntimeError(f"Segmentator failed: {result.stderr}")

    if not expected_output.exists():
        # Try to find any .segs.json in the directory
        segs_files = list(ply_path.parent.glob("*.segs.json"))
        if segs_files:
            expected_output = segs_files[0]
        else:
            raise RuntimeError(f"Segmentator did not produce expected output {expected_output}")

    logger.info("Superpoints saved to %s", expected_output)
    return expected_output


def _ensure_segmentator_ply(ply_path: Path) -> Path:
    """Ensure PLY uses float32 verts and uint32 face indices."""
    try:
        ply = PlyData.read(str(ply_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read PLY {ply_path}: {exc}")

    if "vertex" not in ply or "face" not in ply:
        raise RuntimeError(f"PLY missing vertex/face elements: {ply_path}")

    v = ply["vertex"].data
    if not all(name in v.dtype.names for name in ("x", "y", "z")):
        raise RuntimeError(f"PLY vertex missing x/y/z: {ply_path}")

    verts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    verts_ok = verts.dtype == np.float32

    f = ply["face"].data
    face_prop = None
    for name in ("vertex_indices", "vertex_index"):
        if name in f.dtype.names:
            face_prop = name
            break
    if face_prop is None:
        raise RuntimeError(f"PLY face missing vertex_indices: {ply_path}")

    faces = np.vstack(f[face_prop])
    faces_ok = faces.dtype == np.uint32 and faces.shape[1] == 3

    if verts_ok and faces_ok and face_prop == "vertex_indices":
        return ply_path

    fixed_path = ply_path.with_suffix("")
    fixed_path = fixed_path.with_name(f"{fixed_path.name}.segmentator.ply")

    verts = verts.astype(np.float32)
    faces = faces[:, :3].astype(np.uint32)

    verts_el = np.empty(verts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts_el["x"], verts_el["y"], verts_el["z"] = verts[:, 0], verts[:, 1], verts[:, 2]

    face_el = np.empty(faces.shape[0], dtype=[("vertex_indices", "u4", (3,))])
    face_el["vertex_indices"] = faces

    PlyData([PlyElement.describe(verts_el, "vertex"), PlyElement.describe(face_el, "face")], text=False).write(str(fixed_path))
    logger.info("Wrote Segmentator-friendly PLY to %s", fixed_path)
    return fixed_path


# ---------------------------------------------------------------------------
# Custom SAI3DBase subclass for our pipeline
# ---------------------------------------------------------------------------


def _make_args_namespace() -> types.SimpleNamespace:
    """Create the args namespace expected by SAI3DBase."""
    return types.SimpleNamespace(
        max_neighbor_distance=MAX_NEIGHBOR_DISTANCE,
        similar_metric=SIMILAR_METRIC,
        view_freq=VIEW_FREQ,
        dis_decay=DIS_DECAY,
        use_torch=True,
        thres_trunc=THRES_TRUNC,
        thres_merge=THRES_MERGE,
        from_points_thres=FROM_POINTS_THRES,
        scannetpp=False,
    )


class PipelineSAI3D:
    """Wraps SAI3DBase to work with our observation-based pipeline.

    Instead of loading from ScanNet directories, we feed data directly.
    """

    def __init__(
        self,
        mesh_vertices: np.ndarray,
        seg_ids: np.ndarray,
        masks: np.ndarray,
        depths: np.ndarray,
        poses: np.ndarray,
        intrinsics: np.ndarray,
        color_h: int,
        color_w: int,
        depth_h: int,
        depth_w: int,
    ):
        from sai3d_base import SAI3DBase
        from sai3d import ScanNet_SAI3D

        args = _make_args_namespace()
        self.base = SAI3DBase(mesh_vertices.astype(np.float32), args)

        seg_ids_source = seg_ids

        def _get_seg_data(self_ref, **kwargs):
            if kwargs.get("seg_ids") is None and kwargs.get("points_obj_labels_path") is None:
                kwargs["seg_ids"] = seg_ids_source
            return ScanNet_SAI3D.get_seg_data(self_ref, **kwargs)

        self.base.get_seg_data = types.MethodType(_get_seg_data, self.base)

        # Set up all the attributes SAI3DBase.assign_label() needs
        self.base.masks = masks  # (M, H, W)
        self.base.depths = depths  # (M, H, W)
        self.base.poses = poses  # (M, 4, 4)
        self.base.color_intrinsics = intrinsics  # (M, 3, 3)
        self.base.depth_intrinsics = intrinsics  # same camera for color and depth
        self.base.M = masks.shape[0]
        self.base.CH = color_h
        self.base.CW = color_w
        self.base.DH = depth_h
        self.base.DW = depth_w
        self.base.base_dir = ""
        self.base.scene_id = ""
        self.base.scannetpp = False
        self._seg_ids = seg_ids
        self._scannet_seg_ids = None

    def run(self) -> np.ndarray:
        """Run the full SAI3D pipeline and return per-vertex labels."""
        thres_connect = np.linspace(THRES_CONNECT_START, THRES_CONNECT_END, THRES_CONNECT_STAGES)

        if self._scannet_seg_ids is None:
            seg_ids, seg_num, seg_members, seg_indirect_neighbors = self.base.get_seg_data(
                base_dir="",
                scene_id="",
                max_neighbor_distance=MAX_NEIGHBOR_DISTANCE,
                seg_ids=self._seg_ids,
            )
            self._scannet_seg_ids = (seg_ids, seg_num, seg_members, seg_indirect_neighbors)
        else:
            seg_ids, seg_num, seg_members, seg_indirect_neighbors = self._scannet_seg_ids
        self.base.seg_ids = seg_ids
        self.base.seg_num = seg_num
        self.base.seg_members = seg_members
        self.base.seg_indirect_neighbors = seg_indirect_neighbors

        labels = self.base.assign_label(
            self.base.points,
            thres_connect=thres_connect,
            vis_dis=VIS_DIS,
            max_neighbor_distance=MAX_NEIGHBOR_DISTANCE,
            similar_metric=SIMILAR_METRIC,
        )
        return labels


# ---------------------------------------------------------------------------
# Vertex-label → per-frame pixel mask back-projection
# ---------------------------------------------------------------------------


def _vertex_labels_to_pixel_masks(
    vertices: np.ndarray,
    labels: np.ndarray,
    frames: List[Frame],
) -> Dict[str, np.ndarray]:
    """Project mesh vertex labels onto each frame to produce per-pixel instance masks.

    For each frame we:
    1. Transform vertices to camera coords via X_VW_opencv.
    2. Project to pixel coords via intrinsics.
    3. Keep only vertices in front of camera and within image bounds.
    4. Use a z-buffer to assign the label of the closest vertex to each pixel.

    Returns dict mapping frame.name → (H, W) int32 array of instance labels.
    """
    result: Dict[str, np.ndarray] = {}

    # Pre-compute homogeneous vertices
    N = vertices.shape[0]
    verts_homo = np.hstack([vertices, np.ones((N, 1), dtype=np.float32)])  # (N, 4)

    for frame in tqdm(frames, desc="Back-projecting vertex labels to pixels"):
        H, W = frame.h, frame.w
        extrinsic = frame.X_VW_opencv.cpu().numpy()  # (4, 4)
        K = frame.K.cpu().numpy()  # (3, 3)

        # Transform to camera coordinates
        pts_cam = (extrinsic @ verts_homo.T).T  # (N, 4)
        pts_cam = pts_cam[:, :3]  # (N, 3)

        # Filter points behind camera
        valid = pts_cam[:, 2] > 0
        pts_cam_valid = pts_cam[valid]
        labels_valid = labels[valid]

        # Project to pixel coords
        pts_proj = (K @ pts_cam_valid.T).T  # (N', 3)
        z = pts_proj[:, 2]
        u = np.round(pts_proj[:, 0] / np.clip(z, 1e-8, None)).astype(np.int32)
        v = np.round(pts_proj[:, 1] / np.clip(z, 1e-8, None)).astype(np.int32)

        # Filter to within image bounds
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[in_bounds]
        v = v[in_bounds]
        z_valid = z[in_bounds]
        labels_px = labels_valid[in_bounds]

        # Z-buffer: for each pixel, keep the closest vertex's label
        pixel_mask = np.zeros((H, W), dtype=np.int32)
        z_buffer = np.full((H, W), np.inf, dtype=np.float32)

        for idx in range(len(u)):
            if z_valid[idx] < z_buffer[v[idx], u[idx]]:
                z_buffer[v[idx], u[idx]] = z_valid[idx]
                pixel_mask[v[idx], u[idx]] = labels_px[idx]

        result[frame.name] = pixel_mask

    return result


def _vertex_labels_to_pixel_masks_vectorized(
    vertices: np.ndarray,
    labels: np.ndarray,
    frames: List[Frame],
) -> Dict[str, np.ndarray]:
    """Vectorized version of vertex-label to pixel mask projection.

    Uses numpy vectorized operations with scatter-style z-buffer for speed.
    """
    result: Dict[str, np.ndarray] = {}
    N = vertices.shape[0]
    verts_homo = np.hstack([vertices, np.ones((N, 1), dtype=np.float32)])

    for frame in tqdm(frames, desc="Back-projecting vertex labels to pixels"):
        H, W = frame.h, frame.w
        extrinsic = frame.X_VW_opencv.cpu().numpy()
        K = frame.K.cpu().numpy()

        # Transform to camera coordinates
        pts_cam = (extrinsic @ verts_homo.T).T[:, :3]

        # Filter points behind camera
        valid = pts_cam[:, 2] > 0
        pts_cam_valid = pts_cam[valid]
        labels_valid = labels[valid]

        # Project to pixel coords
        pts_proj = (K @ pts_cam_valid.T).T
        z = pts_proj[:, 2]
        u = np.round(pts_proj[:, 0] / np.clip(z, 1e-8, None)).astype(np.int32)
        v = np.round(pts_proj[:, 1] / np.clip(z, 1e-8, None)).astype(np.int32)

        # Filter to within image bounds
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[in_bounds]
        v = v[in_bounds]
        z_valid = z[in_bounds]
        labels_px = labels_valid[in_bounds]

        # Sort by depth (farthest first) so closer points overwrite
        sort_idx = np.argsort(-z_valid)
        u = u[sort_idx]
        v = v[sort_idx]
        labels_px = labels_px[sort_idx]

        # Paint: later (closer) points overwrite earlier (farther) points
        pixel_mask = np.zeros((H, W), dtype=np.int32)
        pixel_mask[v, u] = labels_px.astype(np.int32)

        result[frame.name] = pixel_mask

    return result


def _encode_ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    """Encode integer ids into RGB colors (0-255) without collision."""
    ids = ids.astype(np.int64)
    ids = np.clip(ids, 0, 0xFFFFFF)
    r = (ids >> 16) & 0xFF
    g = (ids >> 8) & 0xFF
    b = ids & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _triangle_labels_from_vertices(mesh: o3d.geometry.TriangleMesh, labels: np.ndarray) -> np.ndarray:
    triangles = np.asarray(mesh.triangles)
    tri_labels = labels[triangles]
    a = tri_labels[:, 0]
    b = tri_labels[:, 1]
    c = tri_labels[:, 2]
    return np.where((a == b) | (a == c), a, np.where(b == c, b, a))


def _mesh_with_per_triangle_vertex_colors(
    mesh: o3d.geometry.TriangleMesh,
    tri_labels: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    tri_vertices = vertices[triangles]
    new_vertices = tri_vertices.reshape(-1, 3)
    new_triangles = np.arange(new_vertices.shape[0]).reshape(-1, 3)

    tri_colors = _encode_ids_to_rgb(tri_labels).astype(np.float32) / 255.0
    new_colors = np.repeat(tri_colors, 3, axis=0)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    return new_mesh


def _render_instance_id_masks(
    mesh: o3d.geometry.TriangleMesh,
    labels: np.ndarray,
    frames: List[Frame],
) -> Optional[Dict[str, np.ndarray]]:
    """Render per-frame instance id masks with proper occlusion via raycasting."""

    def _to_legacy_mesh(input_mesh):
        if isinstance(input_mesh, o3d.geometry.TriangleMesh):
            return input_mesh
        if hasattr(input_mesh, "to_legacy"):
            try:
                return input_mesh.to_legacy()
            except Exception:
                pass
        legacy = o3d.geometry.TriangleMesh()
        legacy.vertices = o3d.utility.Vector3dVector(np.asarray(input_mesh.vertices))
        legacy.triangles = o3d.utility.Vector3iVector(np.asarray(input_mesh.triangles))
        return legacy

    def _to_tensor_mesh(input_mesh: o3d.geometry.TriangleMesh) -> "o3d.t.geometry.TriangleMesh":
        return o3d.t.geometry.TriangleMesh.from_legacy(input_mesh)

    def _raycast_instance_id_masks(
        mesh_legacy: o3d.geometry.TriangleMesh,
        tri_labels: np.ndarray,
        frames: List[Frame],
    ) -> Dict[str, np.ndarray]:
        scene = o3d.t.geometry.RaycastingScene()
        tmesh = _to_tensor_mesh(mesh_legacy)
        scene.add_triangles(tmesh)

        result: Dict[str, np.ndarray] = {}
        for frame in tqdm(frames, desc="Raycasting instance id masks"):
            H, W = frame.h, frame.w
            K = frame.K.cpu().numpy()
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

            u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            u = u + 0.5
            v = v + 0.5
            dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
            dirs_cam = dirs_cam.reshape(-1, 3)
            dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

            X_VW = frame.X_VW_opencv.cpu().numpy()
            X_WV = np.linalg.inv(X_VW)
            R = X_WV[:3, :3]
            t = X_WV[:3, 3]
            dirs_world = dirs_cam @ R.T
            origins = np.broadcast_to(t, dirs_world.shape)

            rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
            ans = scene.cast_rays(o3d.core.Tensor(rays))
            prim_ids = ans["primitive_ids"].numpy().reshape(H, W)

            mask = np.zeros((H, W), dtype=np.int32)
            if np.issubdtype(prim_ids.dtype, np.unsignedinteger):
                invalid = np.iinfo(prim_ids.dtype).max
                hit = prim_ids != invalid
            else:
                hit = prim_ids >= 0
            if np.any(hit):
                prim_ids_valid = prim_ids[hit].astype(np.int64)
                mask[hit] = tri_labels[prim_ids_valid]
            result[frame.name] = mask

        return result

    mesh_legacy = _to_legacy_mesh(copy.deepcopy(mesh))
    tri_labels = _triangle_labels_from_vertices(mesh_legacy, labels)
    return _raycast_instance_id_masks(mesh_legacy, tri_labels, frames)


# ---------------------------------------------------------------------------
# Workspace filtering of vertex labels
# ---------------------------------------------------------------------------


def _filter_labels_by_workspace(
    vertices: np.ndarray,
    labels: np.ndarray,
    workspace_voxels: o3d.geometry.VoxelGrid,
) -> np.ndarray:
    """Zero out labels for segments where <90% of points are in workspace."""
    labels = labels.copy()
    pcd = o3d.utility.Vector3dVector(vertices)
    valid_mask = np.array(workspace_voxels.check_if_included(pcd))

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl <= 0:
            continue
        seg_mask = labels == lbl
        total = np.sum(seg_mask)
        inside = np.sum(valid_mask & seg_mask)
        if total > 0 and inside / total < 0.9:
            labels[seg_mask] = 0

    return labels


def _cull_mesh_by_workspace(
    mesh: o3d.geometry.TriangleMesh,
    workspace_voxels: o3d.geometry.VoxelGrid,
) -> o3d.geometry.TriangleMesh:
    """Cull mesh vertices/triangles to workspace voxels only."""
    mesh = copy.deepcopy(mesh)
    vertices = np.asarray(mesh.vertices)
    valid_mask = np.array(workspace_voxels.check_if_included(o3d.utility.Vector3dVector(vertices)))
    if not valid_mask.all():
        mesh.remove_vertices_by_mask(~valid_mask)
    return mesh


def _crop_mesh_to_workspace_bbox(
    mesh: o3d.geometry.TriangleMesh,
    workspace_voxels: o3d.geometry.VoxelGrid,
    padding_m: float = 0.2,
) -> o3d.geometry.TriangleMesh:
    """Crop mesh to workspace voxel bounding box."""
    voxel_size = float(workspace_voxels.voxel_size)
    origin = np.asarray(workspace_voxels.origin, dtype=np.float32)
    voxels = workspace_voxels.get_voxels()
    if len(voxels) == 0:
        return mesh

    indices = np.array([v.grid_index for v in voxels], dtype=np.float32)
    min_corner = origin + indices.min(axis=0) * voxel_size - padding_m
    max_corner = origin + (indices.max(axis=0) + 1.0) * voxel_size + padding_m
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
    return mesh.crop(aabb)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def initialize_scene(
    observations: Observations,
    scene: SceneSetup,
    intermediate_outputs_path: Optional[Path] = None,
) -> ObjectSegmentations:
    """Run the full SAI3D pipeline on the given observations.

    Steps:
    1. Convert ObservationFrames → psdframe.Frame
    2. TSDF mesh reconstruction
    3. Export mesh as PLY → run Segmentator → segs.json
    4. Generate Semantic-SAM 2D masks per frame
    5. Build workspace voxel grid
    6. Run SAI3D progressive region growing
    7. Filter labels by workspace, remove table
    8. Convert vertex labels → per-frame pixel masks
    9. Return ObjectSegmentations
    """

    if intermediate_outputs_path is not None:
        intermediate_outputs_path.mkdir(parents=True, exist_ok=True)

    # Debug visualizer (activated by SAI3D_DEBUG=1 env var)
    dbg_dir = (intermediate_outputs_path if intermediate_outputs_path is not None else Path("/tmp/sai3d_work")) / "debug"
    dbg = DebugVisualizer(dbg_dir)

    # ---- 1. Convert frames ----
    logger.info("Converting %d observation frames …", len(observations.frames))
    frames = [get_dataset_frame_from_observation_frame(f) for f in observations.frames]
    dbg.save_frames(frames)

    # ---- 2. TSDF mesh reconstruction ----
    logger.info("Reconstructing TSDF mesh …")
    mesh = _extract_mesh_bounded_with_res(frames, depth_trunc=2, mesh_res=1024)
    mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
    logger.info("Mesh has %d vertices, %d triangles", len(mesh.vertices), len(mesh.triangles))
    dbg.save_mesh(mesh)

    # ---- 3. Workspace voxel grid ----
    workspace_voxels = get_workspace_voxels(scene)
    dbg.save_workspace_voxels(workspace_voxels)

    # Crop mesh to workspace bounding box before Segmentator
    mesh = _crop_mesh_to_workspace_bbox(mesh, workspace_voxels)
    mesh_vertices = np.asarray(mesh.vertices).astype(np.float32)
    logger.info("Cropped mesh has %d vertices, %d triangles", len(mesh.vertices), len(mesh.triangles))
    dbg.save_mesh(mesh)

    # ---- 4. Export PLY + run Segmentator ----
    work_dir = intermediate_outputs_path if intermediate_outputs_path is not None else Path("/tmp/sai3d_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    ply_path = work_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh)
    logger.info("Mesh exported to %s", ply_path)

    segs_json_path = _run_segmentator(ply_path)
    with open(segs_json_path, "r") as f:
        seg_data = json.load(f)
    seg_ids = np.array(seg_data["segIndices"], dtype=np.int32)
    logger.info("Loaded %d superpoint assignments from %s", len(seg_ids), segs_json_path)
    dbg.save_superpoints(mesh, seg_ids)
    del seg_data

    # ---- 5. Semantic-SAM 2D masks ----
    logger.info("Generating Semantic-SAM 2D masks …")
    mask_generator = _build_mask_generator()
    mask_cache_dir = work_dir / "semantic_sam_masks" if intermediate_outputs_path is not None else None

    all_masks = []
    all_depths = []
    all_poses = []
    all_intrinsics = []

    for frame in tqdm(frames, desc="Semantic-SAM mask generation"):
        mask_2d = _generate_masks_for_frame(frame, mask_generator, cache_dir=mask_cache_dir)
        all_masks.append(mask_2d)

        assert frame.depth is not None
        all_depths.append(frame.depth.cpu().numpy())

        # SAI3D expects cam-to-world poses (i.e. X_WV), but in OpenCV convention.
        # sai3d_base uses inv(pose) to go world→cam, so pose should be cam→world.
        # X_WV_opencv = X_WV @ x_rot_flip  → we need OpenCV-convention cam-to-world
        pose = frame.X_WV_opencv.cpu().numpy()
        all_poses.append(pose)

        K = frame.K.cpu().numpy()
        all_intrinsics.append(K)

    masks_np = np.stack(all_masks, axis=0)  # (M, H, W)
    depths_np = np.stack(all_depths, axis=0)  # (M, H, W)
    poses_np = np.stack(all_poses, axis=0).astype(np.float32)  # (M, 4, 4)
    intrinsics_np = np.stack(all_intrinsics, axis=0).astype(np.float32)  # (M, 3, 3)
    del all_masks, all_depths, all_poses, all_intrinsics
    del mask_generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    color_h, color_w = masks_np.shape[1], masks_np.shape[2]
    depth_h, depth_w = depths_np.shape[1], depths_np.shape[2]

    logger.info("Masks shape: %s, Depths shape: %s", masks_np.shape, depths_np.shape)
    dbg.save_masks_2d(frames, masks_np)

    # ---- 6. Run SAI3D ----
    logger.info("Running SAI3D progressive region growing …")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pipeline = PipelineSAI3D(
        mesh_vertices=mesh_vertices,
        seg_ids=seg_ids,
        masks=masks_np,
        depths=depths_np,
        poses=poses_np,
        intrinsics=intrinsics_np,
        color_h=color_h,
        color_w=color_w,
        depth_h=depth_h,
        depth_w=depth_w,
    )
    vertex_labels = pipeline.run()
    logger.info("SAI3D produced %d unique labels", len(np.unique(vertex_labels)))
    dbg.save_segmented_mesh(mesh, vertex_labels)

    # ---- 7. Filter by workspace + remove table ----
    vertex_labels_raw = vertex_labels.copy()  # save pre-filter copy for debug comparison
    vertex_labels = _filter_labels_by_workspace(mesh_vertices, vertex_labels, workspace_voxels)

    # Convert vertex labels to per-frame pixel masks for table detection
    instance_groups = _render_instance_id_masks(mesh, vertex_labels, frames)
    if instance_groups is None:
        raise RuntimeError("instance grouping failed")
        # instance_groups = _vertex_labels_to_pixel_masks_vectorized(mesh_vertices, vertex_labels, frames)

    # Determine valid object IDs (appear in at least 3 frames)
    all_label_ids = np.unique(vertex_labels)
    all_label_ids = all_label_ids[all_label_ids > 0]

    frame_counts = {lbl: 0 for lbl in all_label_ids}
    for frame_name, mask in instance_groups.items():
        for lbl in all_label_ids:
            if np.any(mask == lbl):
                frame_counts[lbl] += 1

    valid_ids = np.array([lbl for lbl, cnt in frame_counts.items() if cnt >= 3])
    logger.info("Labels in >= 3 frames: %d / %d", len(valid_ids), len(all_label_ids))

    # Zero out invalid IDs in masks
    for name in instance_groups:
        instance_groups[name][~np.isin(instance_groups[name], valid_ids)] = 0

    # Remove table
    table_id = determine_table_instance_id(frames, instance_groups, scene.ground_plane, valid_ids)
    logger.info("Table instance id: %d", table_id)
    if table_id > 0:
        valid_ids = valid_ids[valid_ids != table_id]
        for name in instance_groups:
            instance_groups[name][instance_groups[name] == table_id] = 0

    # Build final vertex labels for debug (apply same filtering to vertex array)
    vertex_labels_filtered = vertex_labels.copy()
    vertex_labels_filtered[~np.isin(vertex_labels_filtered, valid_ids)] = 0
    dbg.save_filtered_mesh(mesh, vertex_labels_raw, vertex_labels_filtered, table_id, valid_ids)

    # ---- 8. Build InstanceMaskObjectsDef ----
    frame_ids: List[int] = []
    pixel_masks: List[np.ndarray] = []

    results_path = None
    if intermediate_outputs_path is not None:
        results_path = intermediate_outputs_path / "instances"
        results_path.mkdir(parents=True, exist_ok=True)

    for obs_frame in observations.frames:
        frame_ids.append(obs_frame.id)
        mask = instance_groups.get(obs_frame.name, np.zeros((frames[0].h, frames[0].w), dtype=np.int32))
        pixel_masks.append(mask)

    instance_mask_objects = InstanceMaskObjectsDef(
        frame_ids=frame_ids,
        pixel_object_ids=pixel_masks,
    )

    logger.info("Initialized %d objects (after table removal)", len(valid_ids))
    dbg.save_pixel_masks(frames, instance_groups)
    return ObjectSegmentations(object_segmentations=instance_mask_objects)
