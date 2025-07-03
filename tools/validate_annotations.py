import logging
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

from src.utils import read_complete_annotation_file
from src.utils import validate_annotations
from src.utils import write_annotation_file

logger = logging.getLogger("VALIDATOR")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="validate_annotations"
)  # fmt: off
def run_validation(cfg: DictConfig):
    """
    Run the annotation validation process.

    For each panel in each image, show its bounding box and the
    detected number. Prompt the user if this particular annotation
    should be kept, and write a validated annotation file.

    Note: false annotations are simply discarded (i.e. not corrected)

    Parameters
    ----------
    cfg: DictConfig
        the config

    """
    logger.setLevel(level=logging.DEBUG)

    # image files
    input_images_dir = Path(cfg.input_images_dir)
    image_files = [
        i
        for i in input_images_dir.iterdir()
        if i.suffix in cfg.image_file_extensions
    ]
    image_files.sort()

    # validated annotations dir
    validated_annotations_dir = Path(cfg.output_dir)
    try:
        validated_annotations_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.error(f"{validated_annotations_dir} already exists")
        import sys

        sys.exit()

    # stats on annotations
    n_total_annotations = 0
    n_remaining_annotations = 0

    # let's go !
    for count, image_file in enumerate(image_files):
        annotation_filename = image_file.stem
        annotation_filename = Path(annotation_filename).with_suffix(".csv")
        annotation_filename = (
            Path(cfg.input_annotations_dir) / annotation_filename
        )
        if not annotation_filename.is_file():
            logger.debug(f"No annotations for {image_file.stem}")
            continue

        logger.debug(
            f"Processing {image_file.stem} ({count + 1}/{len(image_files)})"
        )

        # read annotation file
        annotations = read_complete_annotation_file(annotation_filename)
        n_total_annotations += len(annotations)

        # load image
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # show annotations and validate
        final_annotations = validate_annotations(image, annotations, show=True)

        # save final annotations
        numbers = []
        boxes = []

        if len(final_annotations) > 0:
            n_remaining_annotations += len(final_annotations)
            for a in final_annotations:
                numbers.append(a[0])
                boxes.append(a[1])

            validated_annotation_filename = image_file.stem
            validated_annotation_filename = Path(
                validated_annotation_filename
            ).with_suffix(".csv")
            validated_annotation_filename = (
                validated_annotations_dir / validated_annotation_filename
            )
            write_annotation_file(
                validated_annotation_filename, boxes, numbers
            )
        else:
            logger.debug("No annotations for this image !")

    logger.debug(
        f"Total number of detected annotations: {n_total_annotations}"
    )
    logger.debug(f"Number of remaining annotations: {n_remaining_annotations}")


if __name__ == "__main__":
    run_validation()
