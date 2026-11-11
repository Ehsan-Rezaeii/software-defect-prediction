import csv
import io

import numpy as np
import pytest

from software_defect_prediction.data import HEADER, groups, parse, verified


def raw(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(HEADER)
    writer.writerows(rows)
    return stream.getvalue().encode()


def test_positional_name_and_positive_class():
    data = parse(raw([["camel", "1.4", "ClassA", *range(20), 2]]), "1.4")
    assert data.ids == ("ClassA",)
    assert data.y.tolist() == [1]
    assert data.x.shape == (1, 20)


@pytest.mark.parametrize("bug", [-1, 0.5, "nan"])
def test_bad_bug(bug):
    with pytest.raises(ValueError):
        parse(raw([["camel", "1.4", "ClassA", *range(20), bug]]), "1.4")


def test_header_missing_nonfinite_and_identity():
    for value in [b"name,bug\n", b"", raw([["camel", "1.4", "A", *(["nan"] * 20), 0]])]:
        with pytest.raises(ValueError):
            parse(value, "1.4")
    with pytest.raises(ValueError, match="identity"):
        verified(b"invalid", "1.4")


def test_full_vector_groups():
    x = np.array([[1, 2], [1, 2], [1, 3]])
    assert groups(x).tolist() == [0, 0, 1]
