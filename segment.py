"""CLI entry point for SAI3D external segmenter.

Usage (invoked by pixi task):
    python segment.py <observations_path> <scene_path>

Outputs ``objects_path: <path>`` to stdout for the parent process to read.
"""

from dataclasses import dataclass
from pathlib import Path
import contextlib
import logging
import random
import sys
import time
import numpy as np
import torch
import tyro

from initializerdefs import Observations, SceneSetup, get_mesh_path_for_transforms, load_observations_from_transforms_path
from segmenter import initialize_scene


DEFAULT_SEED = 0


@dataclass
class Args:
    transforms_path: tyro.conf.Positional[Path]
    """Path to transforms.json for the dataset."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled SceneSetup."""

    with_workspace_mask_filter: bool = False
    """Enable per-frame 2D workspace-overlap mask filtering before SAI3D."""


def run() -> None:
    logger = logging.getLogger("sai3d-segmenter")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    args = tyro.cli(Args)

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    with contextlib.redirect_stdout(sys.stderr):
        logger.info("--------------")
        logger.info("Starting SAI3D initialization")
        logger.info("params: %s", args)
        logger.info("Determinism enabled with seed=%d", DEFAULT_SEED)
        logger.info("2D workspace mask filter enabled: %s", args.with_workspace_mask_filter)

        logger.info("Loading observations from %s …", args.transforms_path)
        dataset: Observations = load_observations_from_transforms_path(args.transforms_path)
        logger.info("Observations loaded.")

        logger.info("Loading scene setup from %s …", args.scene_path)
        scene = SceneSetup.load(args.scene_path)
        logger.info("Scene loaded.")

        if dataset.id is None:
            logger.info("Dataset has no id, using transient id")
            dataset.id = f"transient_{time.strftime('%Y%m%d-%H%M%S')}"

        project_root = Path(__file__).parent
        output_dir = project_root / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{dataset.id}"

        logger.info("Initializing scene …")
        objects = initialize_scene(
            dataset,
            scene,
            intermediate_outputs_path=output_dir,
            with_workspace_mask_filter=args.with_workspace_mask_filter,
            mesh_path=get_mesh_path_for_transforms(args.transforms_path),
        )

        output_path = output_dir / "objectsdef.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        objects.save(output_path)

        logger.info("Objects saved to %s", output_path)

    # Output in format expected by ExternalSegmentationInitializer
    print(f"objects_path: {output_path}")


if __name__ == "__main__":
    run()
