#!/usr/bin/env python3
"""
A configured ``database_type`` overrides detection, and a ``.parquet`` output holds exactly
what the CSV output holds.

Runs on the committed Dexcom Clarity fixture, which the detector reads as native ``dexcom``
and the override sends through the cgm_format backend instead.
"""

from pathlib import Path
from typing import List

import polars as pl
import pytest
import yaml
from polars.testing import assert_frame_equal

from formats.cgm_format_input.cgm_format_database_converter import CGM_FORMAT_DATABASE_TYPE
from formats.database_detector import DatabaseDetector
from glucose_ml_preprocessor import PARTIAL_CSV_SUFFIX, STATS_SUFFIX, GlucoseMLPreprocessor

REPO_ROOT = Path(__file__).parent.parent
DEXCOM_SMALL = REPO_ROOT / "test_data" / "dexcom_small"
LIVIA_CONFIG = REPO_ROOT / "glucose_config_livia.yaml"


def _preprocessor(**overrides: object) -> GlucoseMLPreprocessor:
    return GlucoseMLPreprocessor.from_config_file(LIVIA_CONFIG, print_statistics=False, max_workers=1, **overrides)


def _process(output_file: Path, inputs: List[Path]) -> None:
    preprocessor = _preprocessor()
    if len(inputs) == 1:
        preprocessor.process(inputs[0], output_file)
    else:
        preprocessor.process_multiple_databases(inputs, output_file)


def test_configured_database_type_overrides_detection() -> None:
    detected = DatabaseDetector().detect_database_type(DEXCOM_SMALL)
    assert detected != CGM_FORMAT_DATABASE_TYPE
    with open(LIVIA_CONFIG, encoding="utf-8") as f:
        configured = yaml.safe_load(f)["database_type"]
    assert configured == CGM_FORMAT_DATABASE_TYPE

    assert _preprocessor().resolve_database_type(DEXCOM_SMALL) == configured
    assert _preprocessor(database_type=detected).resolve_database_type(DEXCOM_SMALL) == detected
    assert GlucoseMLPreprocessor(database_type=None).resolve_database_type(DEXCOM_SMALL) == detected


def test_unknown_database_type_is_refused() -> None:
    with pytest.raises(ValueError, match="no_such_format"):
        _preprocessor(database_type="no_such_format").resolve_database_type(DEXCOM_SMALL)


@pytest.mark.parametrize("inputs", [[DEXCOM_SMALL], [DEXCOM_SMALL, DEXCOM_SMALL]], ids=["single", "multiple"])
def test_parquet_output_matches_csv_output(tmp_path: Path, inputs: List[Path]) -> None:
    csv_file = tmp_path / "csv" / "out.csv"
    parquet_file = tmp_path / "parquet" / "out.parquet"
    csv_file.parent.mkdir()
    parquet_file.parent.mkdir()

    _process(csv_file, inputs)
    _process(parquet_file, inputs)

    from_parquet = pl.read_parquet(parquet_file)
    from_csv = pl.read_csv(csv_file, infer_schema_length=None, try_parse_dates=True)
    assert from_csv.height > 0
    assert_frame_equal(from_parquet, from_csv)
    assert from_parquet.schema["timestamp"] == pl.Datetime("us")
    for name in ("glucose_value_mgdl", "fast_acting_insulin_u", "long_acting_insulin_u", "carb_grams"):
        assert from_parquet.schema[name] == pl.Float64, name

    # The working CSV is gone; the statistics sit beside the parquet as they do beside a CSV.
    assert sorted(p.name for p in parquet_file.parent.iterdir()) == sorted(
        [parquet_file.name, parquet_file.with_suffix(STATS_SUFFIX).name]
    )
    assert not any(p.name.endswith(PARTIAL_CSV_SUFFIX) for p in parquet_file.parent.iterdir())


def test_appending_to_parquet_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CSV"):
        _preprocessor().process(DEXCOM_SMALL, tmp_path / "out.parquet", append=True)
