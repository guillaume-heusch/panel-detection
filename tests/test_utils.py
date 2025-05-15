import pytest

from src.utils import read_annotation_file
from src.utils import convert_polygons_to_bounding_boxes

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
                [0, 0, 10, 0, 10, 10, 0, 10],      # border
                [50, 50, 60, 50, 60, 60, 50, 60],  # valid
            ],
            100,
            100,
            [
                [10, 10, 20, 20],
                [50, 50, 60, 60],
            ],
        ),
        (
            [],  # Empty input
            100,
            100,
            [],
        ),
    ]
)
def test_convert_polygons_to_bounding_boxes(polygons, height, width, expected):
    assert convert_polygons_to_bounding_boxes(polygons, height, width) == expected

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
    ]
)
def test_malformed_polygons(polygons, height, width, expected_exception):
    with pytest.raises(expected_exception):
        convert_polygons_to_bounding_boxes(polygons, height, width)

