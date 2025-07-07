import logging
from pathlib import Path
from operator import itemgetter

import hydra
from omegaconf import DictConfig
from matplotlib import pyplot as plt

from src.utils import read_complete_annotation_file

logger = logging.getLogger("BROWSE")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="browse_annotations"
)  # fmt: off
def browse(cfg: DictConfig):
    """
    Browse through all annotation files.

    Parameters
    ----------
    cfg: DictConfig
        the config

    """
    logger.setLevel(level=logging.DEBUG)

    # annotation files
    input_dir = Path(cfg.input_annotations_dir)
    annotations_files = [
        i
        for i in input_dir.iterdir()
        if i.suffix in cfg.annotation_file_extension
    ]
    annotations_files.sort()

    # stats on annotations
    n_total_annotations = 0
    smaller_bbox_area = 10000
    larger_bbox_area = 0
    mean_area = 0
    numbers_histogram = {}
    for i in range(1000):
        numbers_histogram[i] = 0

    # let's go !
    for count, annotations_file in enumerate(annotations_files):
        logger.debug(
            f"Processing {annotations_file.stem} "
            f"({count + 1}/{len(annotations_files)})"
        )

        # read annotation file
        annotations = read_complete_annotation_file(annotations_file)
        n_total_annotations += len(annotations)

        for a in annotations:
            numbers_histogram[a[0]] += 1

            b = a[1]
            area = (b[2] - b[0]) * (b[3] - b[1])
            mean_area += area
            if area < smaller_bbox_area:
                smaller_bbox = b
                smaller_bbox_area = area
            if area > larger_bbox_area:
                larger_bbox = b
                larger_bbox_area = area

    f, ax = plt.subplots(1)
    ax.plot(numbers_histogram.keys(), numbers_histogram.values())
    plt.show()

    n = 5
    res = dict(
        sorted(numbers_histogram.items(), key=itemgetter(1), reverse=True)[:n]
    )
    logger.info(f"Total number of annotations: {n_total_annotations}")
    logger.info(f"Most represented numbers: {res}")
    logger.info(
        "Smaller bounding-box: "
        f"{smaller_bbox[2] - smaller_bbox[0]}, "
        f"{smaller_bbox[3] - smaller_bbox[1]} -> {smaller_bbox_area}"
    )
    logger.info(
        "Larger bounding-box: "
        f"{larger_bbox[2] - larger_bbox[0]}, "
        f"{larger_bbox[3] - larger_bbox[1]} -> {larger_bbox_area}"
    )
    mean_area /= n_total_annotations
    logger.info(f"Mean bounding-box area = {mean_area}")


if __name__ == "__main__":
    browse()
