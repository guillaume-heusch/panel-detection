import pytest
import numpy as np
import tempfile
import csv

from pathlib import Path
from unittest.mock import Mock, patch


from src.utils import read_annotation_file
from src.utils import convert_polygons_to_bounding_boxes
from src.utils import read_annotation_file_for_detection
from src.utils import save_show_final_result
from src.utils import review_annotations
from src.utils import read_complete_annotation_file
from src.utils import keep_annotation
from src.utils import validate_annotations
from src.utils import write_annotation_file


def test_read_annotation_file(annotation_file_polygon):
    polygons = read_annotation_file(annotation_file_polygon)
    assert isinstance(polygons, list)
    assert len(polygons) == 16
    assert len(polygons[0]) == 8
    assert isinstance(polygons[0][0], str)


@pytest.mark.parametrize(
    "polygons, height, width, expected",
    [
        (
            [[10, 10, 20, 10, 20, 20, 10, 20]],  # Valid polygon
            100,
            100,
            [[10, 10, 20, 20]],
        ),
        (
            [[0, 0, 10, 0, 10, 10, 0, 10]],  # Touching border
            100,
            100,
            [],
        ),
        (
            [[90, 90, 110, 90, 110, 110, 90, 110]],  # Outside bounds
            100,
            100,
            [],
        ),
        (
            [  # Mix of valid and invalid
                [10, 10, 20, 10, 20, 20, 10, 20],  # valid
                [0, 0, 10, 0, 10, 10, 0, 10],  # border
                [50, 50, 60, 50, 60, 60, 50, 60],  # valid
            ],
            100,
            100,
            [[10, 10, 20, 20], [50, 50, 60, 60]],
        ),
        (
            [],  # Empty input
            100,
            100,
            [],
        ),
    ],
)
def test_convert_polygons_to_bounding_boxes(polygons, height, width, expected):
    assert (
        convert_polygons_to_bounding_boxes(polygons, height, width) == expected
    )


@pytest.mark.parametrize(
    "polygons, height, width, expected_exception",
    [
        (
            "blah",  # polygons not of the right type
            100,
            100,
            TypeError,
        ),
        (
            [[10, 10, 20, 10, 20, 20]],  # Only 6 values, not 8
            100,
            100,
            ValueError,
        ),
        (
            [[10, 10, 20, "a", 20, 20, 10, 20]],  # Non-numeric value
            100,
            100,
            ValueError,
        ),
        (
            [[None, 10, 20, 10, 20, 20, 10, 20]],  #  None is not numeric
            100,
            100,
            ValueError,
        ),
        (
            [[10, 10, 20, 10, 20, 20, 10, float("nan")]],  # NaN value
            100,
            100,
            ValueError,
        ),
    ],
)
def test_malformed_polygons(polygons, height, width, expected_exception):
    with pytest.raises(expected_exception):
        convert_polygons_to_bounding_boxes(polygons, height, width)


def test_read_annotation_file_bbox(annotation_file_bbox):
    targets = read_annotation_file_for_detection(annotation_file_bbox)
    assert isinstance(targets, dict)
    assert "boxes" in targets.keys()
    assert "labels" in targets.keys()
    assert len(targets["boxes"]) == 2
    assert len(targets["labels"]) == 2


def test_save_show_final_result(annotation_file_bbox):
    targets = read_annotation_file_for_detection(annotation_file_bbox)
    boxes = targets["boxes"]
    image = np.random.randint(0, 256, size=(2000, 2000, 3), dtype=np.uint8)
    numbers = ["100", "200"]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        file_path = tmp_path / "test_image.png"
        save_show_final_result(
            image, boxes, numbers, save_filename=str(file_path)
        )
        assert file_path.exists()


def test_read_complete_annotation_file():
    test_data = [
        ["1", "10", "20", "30", "40"],
        ["not_a_number", "15", "25", "35", "45"],
        ["42", "5", "10", "15", "20"],
    ]
    expected_result = [
        (1, [10, 20, 30, 40]),
        (None, [15, 25, 35, 45]),
        (42, [5, 10, 15, 20]),
    ]
    with tempfile.NamedTemporaryFile(
        mode="w+", newline="", delete=False
    ) as tmp:
        writer = csv.writer(tmp)
        writer.writerows(test_data)
        tmp_path = Path(tmp.name)

    result = read_complete_annotation_file(tmp_path)
    assert result == expected_result
    tmp_path.unlink()


def test_review_annotations_keeps_correct_annotations():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotations = [(1, [10, 10, 50, 50]), (2, [60, 60, 90, 90])]
    mock_keep_fn = Mock(side_effect=[False, True])

    result = review_annotations(
        image=image,
        annotations=annotations,
        keep_annotation_fn=mock_keep_fn,
        show=False,
    )
    assert result == [annotations[1]]


def test_keep_annotation(monkeypatch):
    inputs = iter(["maybe", "Y"])  # first invalid, second valid
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert keep_annotation() is True
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert keep_annotation() is False


def test_validate_annotations_valid():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotations = [(42, [10, 10, 50, 50])]
    result = validate_annotations(image, annotations, show=False)
    assert result == annotations


@patch("src.utils.review_annotations")
def test_validate_annotations_suspicious_kept(mock_review):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    suspicious = (1234, [20, 20, 80, 80])
    mock_review.return_value = [suspicious]  # Simulate user kept it
    result = validate_annotations(image, [suspicious], show=False)
    assert result == [suspicious]
    mock_review.assert_called_once()


@patch("src.utils.review_annotations")
def test_validate_annotations_suspicious_removed(mock_review):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    suspicious = (1500, [5, 5, 40, 40])
    mock_review.return_value = []  # Simulate user removed it
    result = validate_annotations(image, [suspicious], show=False)
    assert result == []
    mock_review.assert_called_once()


def test_validate_annotations_none_number():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotations = [(None, [0, 0, 30, 30])]
    result = validate_annotations(image, annotations, show=False)
    assert result == []


def test_write_annotation_file(tmp_path):
    filename = tmp_path / "annotations.csv"
    boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]
    numbers = [123, 456]

    write_annotation_file(str(filename), boxes, numbers)

    with open(filename, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    expected = [
        ["123", "10", "20", "30", "40"],
        ["456", "50", "60", "70", "80"],
    ]
    assert rows == expected
