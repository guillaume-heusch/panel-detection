import csv
import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def read_annotation_file(filename: Path) -> list:
    """
    Read an annotation file containing polygons.

    Returns the list of annotations for polygons

    Parameters
    ----------
    filename: Path

    Returns
    -------
    list:
        The list of polygons

    """
    polygons = []
    with open(filename, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            polygons.append(row)
    return polygons


def read_annotation_file_for_detection(filename: Path) -> dict:
    """
    Read an annotation file.

    The annotation file should contain, in each line,
    a class label and the corresponding bounding box

    Parameters
    ----------
    filename: Path

    Returns
    -------
    dict:
        The "targets" dictionary, containing labels and bounding boxes

    """
    targets = {}
    targets["boxes"] = []
    targets["labels"] = []
    with open(filename, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            targets["labels"].append(int(row[0]))
            box = [int(row[i]) for i in range(1, 5)]
            targets["boxes"].append(box)

    return targets


def convert_polygons_to_bounding_boxes(
        polygons: list,
        height: int,
        width: int,
) -> list:  # fmt: off
    """
    Convert polygons to bounding boxes.

    Polygons are given as [x1, y1, x2, y2, x3, y3, x4, y4]
    Boxes are returned as [xmin, ymin, xmax, ymax]

    Parameters
    ----------
    polygons: list
        The list of polygons
    height: int
        The height of the image
    width: int
        The width of the image

    Returns
    -------
    list:
        The list of bounding boxes

    Raises
    ------
    TypeError
        If a polygon is not a list of coordinates.
    ValueError
        If a polygon does not contain exactly 8 numeric values,
        or contains NaN values.
    """
    boxes = []

    for p in polygons:
        # Validate input
        if not isinstance(p, list):
            raise TypeError("Each polygon must be a list of coordinates.")
        if len(p) != 8:
            raise ValueError(
                f"Polygon must have exactly 8 values, got {len(p)}."
            )
        if not all(isinstance(coord, (int, float)) for coord in p):
            raise ValueError(
                "All polygon coordinates must be numbers (int or float)."
            )
        if any(np.isnan(coord) for coord in p):
            raise ValueError("Polygon contains NaN value(s).")

        xs = [int(p[i]) for i in range(len(p)) if i % 2 == 0]
        ys = [int(p[i]) for i in range(len(p)) if i % 2 == 1]

        left = np.min(xs)
        right = np.max(xs)
        top = np.min(ys)
        bottom = np.max(ys)

        if left > 0 and top > 0 and right < width and bottom < height:
            boxes.append([left, top, right, bottom])
        else:
            logging.debug("box not considered: at the border")

    return boxes


def save_show_final_result(
    image: np.ndarray,
    boxes: np.ndarray,
    numbers: list,
    save_filename: str = None,
    show: bool = False,
    ):  # fmt: off
    """
    Save and / or show the final result.

    Displays the original image with bounding boxes
    around panels, and detected numbers above.

    Parameters
    ----------
    image: np.ndarray
        The image
    boxes: np.ndarray
        The bounding boxes coordinates as a 2-d numpy array
    numbers: list
        The detected numbers
    save_filename: str
        Write the image with detections to the provided filename
    show: bool
        Shows the image with detections

    """
    f, ax = plt.subplots(1, figsize=(16, 9))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for b, n in zip(list(boxes), numbers):
        rect = Rectangle(
            (b[0], b[1]),
            b[2] - b[0],
            b[3] - b[1],
            edgecolor="green",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(b[0], b[1], n, c="limegreen", size="large", weight="bold")

    if save_filename is not None:
        save_path = Path(save_filename)
        parent = save_path.parent.absolute()
        parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
