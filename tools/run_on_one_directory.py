import logging
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

from src.engine.digit_recognition import DigitRecognizer
from src.engine.panel_detector import PanelDetector
from src.utils import save_show_final_result
from src.utils import write_annotation_file

logger = logging.getLogger("PREDICTOR")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run_on_one_dir"
)  # fmt: off
def run_detection(cfg: DictConfig):
    """
    Perform the detection of panels, and get its number.

    Parameters
    ----------
    cfg: DictConfig
        the config

    """
    logger.setLevel(level=logging.DEBUG)

    # engines
    panel_detector = PanelDetector(cfg)
    digit_recognizer = DigitRecognizer()

    # image files
    input_dir = Path(cfg.input_dir)
    image_files = [
        i for i in input_dir.iterdir() if i.suffix in cfg.image_file_extensions
    ]
    image_files.sort()

    # output dir
    annotations_dir = Path(cfg.output_dir)
    try:
        annotations_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.error("Annotations dir already exists")

    # some statistics
    total_number_of_detected_panels = 0
    n_images_with_no_detections = 0

    # let's go !
    for count, image_file in enumerate(image_files):
        annotation_filename = image_file.stem
        annotation_filename = Path(annotation_filename).with_suffix(".csv")
        annotation_filename = Path(cfg.output_dir) / annotation_filename
        if annotation_filename.is_file():
            logger.debug(
                f"annotation for {image_file.stem} exists, skipping !"
            )
            continue

        logger.debug(
            f"Processing {image_file.stem} ({count + 1}/{len(image_files)})"
        )
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # populate the panel_detector.predictions field
        panel_detector.run_on_one_image(image)

        if panel_detector.predictions is None:
            logger.warning("No panels found !")
            n_images_with_no_detections += 1
            continue

        # perform OCR on the detected panels
        panels = panel_detector.get_panels(image)
        total_number_of_detected_panels += 1
        total_number_of_images_with_no_detections
        numbers = digit_recognizer.run_on_all_panels(panels)

        # visualize / save
        predictions = panel_detector.get_predictions()
        boxes = predictions["boxes"]
        boxes = boxes.detach().numpy()
        save_show_final_result(image, boxes, numbers, None, cfg.show_result)
        write_annotation_file(annotation_filename, boxes, numbers)

    logger.debug(
        f"Total number of detected panels: {total_number_of_detected_panels}"
    )
    logger.debug(
        f"Number of images with no detections: {n_images_with_no_detections}"
    )


if __name__ == "__main__":
    run_detection()
