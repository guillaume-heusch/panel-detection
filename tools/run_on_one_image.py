import logging

import cv2
import hydra
from omegaconf import DictConfig

from src.engine.digit_recognition import DigitRecognizer
from src.engine.panel_detector import PanelDetector
from src.utils import save_show_final_result

logger = logging.getLogger("PREDICTOR")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="run_on_one_image"
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

    image = cv2.imread(cfg.image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect panels in the image
    panel_detector = PanelDetector(cfg)
    panel_detector.run_on_one_image(image)
    panels = panel_detector.get_panels(image)

    # perform OCR on the detected panels
    digit_recognizer = DigitRecognizer()
    numbers = digit_recognizer.run_on_all_panels(panels)

    # visualize / save
    predictions = panel_detector.get_predictions()
    boxes = predictions["boxes"]
    boxes = boxes.detach().numpy()
    save_show_final_result(
        image, boxes, numbers, cfg.save_filename, cfg.show_result
    )


if __name__ == "__main__":
    run_detection()
