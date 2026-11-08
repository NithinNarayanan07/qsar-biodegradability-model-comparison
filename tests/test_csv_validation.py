"""Exercise user-controlled test uploads, including pandas' duplicate-header trap."""

from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest

from ml_core import FEATURES, TARGET, read_test_csv, validate_test_data


@pytest.fixture
def valid_frame():
    frame = pd.DataFrame(np.ones((3, len(FEATURES))), columns=FEATURES)
    frame[TARGET] = ["RB", "NRB", "RB"]
    return frame


def test_valid_csv_accepts_bytes_paths_and_file_objects(valid_frame, tmp_path):
    csv = valid_frame.to_csv(index=False)
    path = tmp_path / "test.csv"
    path.write_text(csv, encoding="utf-8")
    for source in [csv.encode(), BytesIO(csv.encode()), StringIO(csv), path]:
        actual = read_test_csv(source)
        assert list(actual.columns) == FEATURES + [TARGET]
        assert actual[TARGET].tolist() == ["RB", "NRB", "RB"]
        assert np.array_equal(actual[FEATURES].to_numpy(), np.ones((3, len(FEATURES))))


def test_reordered_headers_and_whitespace_are_normalized(valid_frame):
    frame = valid_frame.copy()
    frame[TARGET] = [" RB ", "NRB ", " RB"]
    frame = frame[list(reversed(frame.columns))]
    frame.columns = [f" {name} " for name in frame.columns]
    actual = validate_test_data(frame)
    assert list(actual.columns) == FEATURES + [TARGET]
    assert actual[TARGET].tolist() == ["RB", "NRB", "RB"]


@pytest.mark.parametrize("column", [FEATURES[0], TARGET])
def test_missing_feature_or_target_is_rejected(valid_frame, column):
    with pytest.raises(ValueError):
        validate_test_data(valid_frame.drop(columns=column))


def test_extra_column_is_rejected(valid_frame):
    with pytest.raises(ValueError):
        validate_test_data(valid_frame.assign(accidental_index=[0, 1, 2]))


def test_duplicate_dataframe_column_is_rejected(valid_frame):
    frame = pd.concat([valid_frame, valid_frame[[FEATURES[0]]]], axis=1)
    with pytest.raises(ValueError):
        validate_test_data(frame)


def test_duplicate_csv_header_is_rejected_before_pandas_can_rename_it(valid_frame):
    csv = valid_frame.to_csv(index=False)
    header, rows = csv.split("\n", 1)
    header = header.replace(FEATURES[1], FEATURES[0], 1)
    with pytest.raises(ValueError):
        read_test_csv((header + "\n" + rows).encode())


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "not-a-number", ""])
def test_missing_nonnumeric_and_nonfinite_features_are_rejected(valid_frame, bad_value):
    frame = valid_frame.astype({FEATURES[0]: object})
    frame.loc[0, FEATURES[0]] = bad_value
    with pytest.raises(ValueError):
        validate_test_data(frame)


@pytest.mark.parametrize("bad_label", ["unknown", "rb", "1", "", None])
def test_unsupported_or_missing_target_is_rejected(valid_frame, bad_label):
    frame = valid_frame.copy()
    frame.loc[0, TARGET] = bad_label
    with pytest.raises(ValueError):
        validate_test_data(frame)


def test_empty_frame_is_rejected(valid_frame):
    with pytest.raises(ValueError):
        validate_test_data(valid_frame.iloc[:0])


@pytest.mark.parametrize("payload", [b"", b"not,a,valid,schema\n1,2,3,4\n"])
def test_empty_or_unrelated_csv_is_rejected(payload):
    with pytest.raises(ValueError):
        read_test_csv(payload)
