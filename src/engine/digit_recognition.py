from typing import Tuple

import numpy as np
from ocr_tamil.ocr import OCR
from omegaconf import DictConfig


class DigitRecognizer:
    """
    Class responsible to perform OCR on detected panels

    User has access to:

    - run_on_panel_image

    Attributes
    ----------

    """

    def __init__(self, cfg: DictConfig = None):
        """

        Parameters
        ----------
        cfg: DictConfig
            The configuration

        """
        self.cfg = cfg
        self.ocr_engine = OCR(details=1, lang=["english"])
        self.ocr_engine.load_model()

    def run_on_all_panels(self, panels: list) -> list:
        """
        Run the OCR on a list of panels

        Parameters
        ----------
        panels: list
            List of panel images as np.ndarray

        Returns
        -------
        list:
            List of detected numbers (string)

        """
        detected_numbers = []
        for panel in panels:
            number, confidence = self.run_on_one_image(panel)
            detected_numbers.append(number)
        return detected_numbers

    def run_on_one_image(self, panel: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on a panel image

        Parameters
        ----------
        panel: np.ndarray
            The image

        Returns
        -------
        str:
            The detected number
        float:
            The confidence of the detection

        """
        number, confidence = self.ocr_engine.text_recognize_batch([panel])
        number = number[0]
        confidence = confidence[0]
        return number, confidence
