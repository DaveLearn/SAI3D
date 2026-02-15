"""CLI entry point for SAI3D external segmenter.

Usage (invoked by pixi task):
    python segment.py <observations_path> <scene_path>

Outputs ``objects_path: <path>`` to stdout for the parent process to read.
"""

from dataclasses import dataclass
from pathlib import Path
import time
import logging
import tyro

from initializerdefs import Observations, SceneSetup
from segmenter import initialize_scene


@dataclass
class Args:
    observations_path: tyro.conf.Positional[Path]
    """Path to the pickled Observations."""

    scene_path: tyro.conf.Positional[Path]
    """Path to the pickled SceneSetup."""


def run() -> None:
    logger = logging.getLogger("sai3d-segmenter")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s"))
    logger.addHandler(ch)

    args = tyro.cli(Args)

    logger.info("--------------")
    logger.info("Starting SAI3D initialization")
    logger.info("params: %s", args)

    logger.info("Loading observations from %s …", args.observations_path)
    dataset: Observations = Observations.load(args.observations_path)
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
    objects = initialize_scene(dataset, scene, intermediate_outputs_path=output_dir)

    output_path = output_dir / "objectsdef.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    objects.save(output_path)

    logger.info("Objects saved to %s", output_path)
    # Output in format expected by ExternalSegmentationInitializer
    print(f"objects_path: {output_path}")


if __name__ == "__main__":
    run()
