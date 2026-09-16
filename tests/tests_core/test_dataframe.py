# SPDX-FileCopyrightText: 2026 Deutsche Telekom AG
#
# SPDX-License-Identifier: Apache-2.0

# type: ignore
import numpy as np
import polars as pl

from datetime import datetime
from polars.testing import assert_frame_equal

from adf.core.dataframe.api import (
    add_aggregates_to_list,
    add_alias_from,
    add_centered_rolling_sum,
    add_date_from_str,
    add_forward_sliding_window,
    add_groupby_first,
    add_groups_from_time_delta_differences,
    add_ones,
    add_replaced_string,
    add_time_delta_difference,
    add_zeros,
    drop_shortest_interval,
    filter_datetime_range,
    filter_group_length,
    generate_windows,
    isin_schema,
    split_intervals,
    split_longest_interval,
)
from adf.core.dataframe.base import (
    create_dataframe,
    create_polars_reader,
    infer_datetime_dtype_for_casting,
    load_dataframe,
    save_dataframe,
    save_dataframe_with_auto_increment,
)


def test_load_dataframe(tmp_path):
    # Create a sample DataFrame and save it as a parquet file
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    file_path = tmp_path / "test.parquet"
    df.write_parquet(file_path)

    # Load the DataFrame using the function
    loaded_df = load_dataframe(file_path)
    print("loaded frame", loaded_df)
    print(type(df))

    # Assert the loaded DataFrame is equal to the original DataFrame
    assert_frame_equal(loaded_df, df)


def test_create_dataframe():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = create_dataframe(data)
    assert df.shape == (3, 2)
    assert df.columns == ["a", "b"]


def test_save_dataframe(tmp_path):
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    file_path = tmp_path / "test.parquet"
    save_dataframe(data, file_path)

    # Load the DataFrame to check if it was saved correctly
    loaded_df = pl.read_parquet(file_path)
    assert loaded_df.shape == (3, 2)
    assert loaded_df.columns == ["a", "b"]


def test_save_dataframe_with_auto_increment(tmp_path):
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    save_dataframe_with_auto_increment(data, tmp_path, "dataset")

    # Check if the file was saved correctly
    saved_files = list(tmp_path.glob("dataset_*.parquet"))
    assert len(saved_files) == 1
    assert saved_files[0].name == "dataset_0.parquet"

    # Save another file and check the increment
    save_dataframe_with_auto_increment(data, tmp_path, "dataset")
    saved_files = list(tmp_path.glob("dataset_*.parquet"))
    assert len(saved_files) == 2
    assert saved_files[1].name == "dataset_1.parquet"


def test_create_polars_reader():
    reader = create_polars_reader("csv")
    assert callable(reader)


def test_infer_datetime_dtype_for_casting():
    schema = {"date": pl.Utf8}
    expr = infer_datetime_dtype_for_casting(schema, "date")
    assert isinstance(expr, pl.Expr)


def test_isin_schema():
    df = pl.DataFrame({"value": [1, 2, 3]})
    assert isin_schema(df, column="value") is True
    assert isin_schema(df, column="non_existent") is False


def test_add_aggregates_to_list():
    df = pl.DataFrame({"group": ["A", "A", "B", "B", "B", "C"], "value": [2, 1, 3, 2, 1, 3]})
    result = add_aggregates_to_list(df, "group", "value").sort("group")
    expected = pl.DataFrame({"group": ["A", "B", "C"], "value": [[2, 1], [3, 2, 1], [3]]})
    assert_frame_equal(result, expected)


def test_add_zeros():
    df = pl.DataFrame({"value": [1, 2, 3]})
    result = add_zeros(df, column="zeros", dtype=pl.Int64)
    expected = pl.DataFrame({"value": [1, 2, 3], "zeros": [0, 0, 0]})
    assert_frame_equal(result, expected)


def test_add_ones():
    df = pl.DataFrame({"value": [1, 2, 3]})
    result = add_ones(df, column="ones", dtype=pl.Int64)
    expected = pl.DataFrame({"value": [1, 2, 3], "ones": [1, 1, 1]})
    assert_frame_equal(result, expected)


def test_add_alias_from():
    df = pl.DataFrame({"value": [1, 2, 3]})
    result = add_alias_from(df, column="value", alias="new_value")
    expected = pl.DataFrame({"value": [1, 2, 3], "new_value": [1, 2, 3]})
    assert_frame_equal(result, expected)


def test_add_date_from_str():
    df = create_dataframe({"start": ["2021-01-01 00:00:00", "2021-01-01 00:00:10"]}, date_columns=["start"])
    result = add_date_from_str(df, column="start")
    expected = pl.DataFrame({"start": [datetime(2021, 1, 1, 0, 0, 0), datetime(2021, 1, 1, 0, 0, 10)]}).with_columns(
        pl.col("start").cast(str).str.to_datetime(time_unit="ns", time_zone=None)
    )
    assert_frame_equal(result, expected)


def test_add_centered_rolling_sum():
    df = create_dataframe(
        {"value": [1, 2, 3], "date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"]},
        date_columns=["date_time"],
    )
    result = add_centered_rolling_sum(df, column="value", safe_range=10)
    expected = pl.DataFrame(
        {
            "value": [1, 2, 3],
            "date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"],
            "count": [3, 5, 3],
        }
    ).with_columns(pl.col("date_time").str.to_datetime(time_unit="ns", time_zone=None))

    assert_frame_equal(result, expected)


def test_add_forward_sliding_window():
    df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
    result = add_forward_sliding_window(df, column="values", window_size=3)
    expected = pl.DataFrame(
        {"values": [1, 2, 3, 4, 5], "values_sliding_window": [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5], [5]]}
    )
    assert_frame_equal(result, expected)


def test_add_groupby_first():
    df = pl.DataFrame({"group": ["A", "A", "B", "B", "B", "C"], "value": [2, 1, 3, 2, 1, 3]})
    result = add_groupby_first(df, sort_cols="value", groupby_cols="group").sort("group")
    expected = pl.DataFrame({"group": ["A", "B", "C"], "value": [2, 3, 3]})
    assert_frame_equal(result, expected)


def test_add_time_delta_difference():
    df = create_dataframe(
        {"date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"]}, date_columns=["date_time"]
    )
    result = add_time_delta_difference(df, column="date_time", out_column="time_diff")
    expected = pl.DataFrame(
        {
            "date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"],
            "time_diff": [None, 10.0, 10.0],
        }
    ).with_columns(pl.col("date_time").str.to_datetime(time_unit="ns", time_zone=None))
    assert_frame_equal(result, expected)


def test_add_groups_from_time_delta_differences():
    df = create_dataframe(
        data={"date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"]},
        date_columns=["date_time"],
    )
    result = add_groups_from_time_delta_differences(df, column="date_time", out_column="time_group", threshold=5)
    expected = pl.DataFrame(
        {"date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"], "time_group": [0, 1, 2]}
    ).with_columns(
        [pl.col("date_time").str.to_datetime(time_unit="ns", time_zone=None), pl.col("time_group").cast(pl.UInt32)]
    )
    assert_frame_equal(result, expected)


def test_add_replaced_string():
    df = pl.DataFrame({"text": ["ab12cd34ef", "gh45ij67kl"]})
    result = add_replaced_string(df, column="text", pattern=r"\d+", replacement="X")
    expected = pl.DataFrame({"text": ["ab12cd34ef", "gh45ij67kl"], "text_mapped": ["abXcdXef", "ghXijXkl"]})
    assert_frame_equal(result, expected)


def test_drop_shortest_interval():
    df = create_dataframe(
        data={
            "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:00", "2021-01-01 00:00:00"],
            "end": ["2021-01-01 00:00:05", "2021-01-01 00:00:10", "2021-01-01 00:00:15"],
            "value": [1, 2, 3],
        },
        date_columns=["start", "end"],
    )
    result = drop_shortest_interval(df, end="end", start="start")
    expected = pl.DataFrame(
        {"start": ["2021-01-01 00:00:00"], "end": ["2021-01-01 00:00:15"], "value": [3]}
    ).with_columns(
        [
            pl.col("start").str.to_datetime(time_unit="ns", time_zone=None),
            pl.col("end").str.to_datetime(time_unit="ns", time_zone=None),
        ]
    )
    assert_frame_equal(result, expected)


def test_split_intervals():
    df = create_dataframe(
        data={
            "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:05"],
            "end": ["2021-01-01 00:00:10", "2021-01-01 00:00:15"],
            "value": [1, 2],
        },
        date_columns=["start", "end"],
    )
    result = split_intervals(df, start="start", end="end", safe_range=5)
    expected = pl.DataFrame(
        {
            "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:05", "2021-01-01 00:00:05", "2021-01-01 00:00:10"],
            "end": ["2021-01-01 00:00:10", "2021-01-01 00:00:10", "2021-01-01 00:00:15", "2021-01-01 00:00:15"],
            "value": [1, 1, 2, 2],
        }
    ).with_columns(
        [
            pl.col("start").str.to_datetime(time_unit="ns", time_zone=None),
            pl.col("end").str.to_datetime(time_unit="ns", time_zone=None),
        ]
    )
    assert_frame_equal(result, expected)


def test_split_longest_interval():
    df = create_dataframe(
        data={
            "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:00"],
            "end": ["2021-01-01 00:00:10", "2021-01-01 00:00:15"],
            "value": [1, 2],
        },
        date_columns=["start", "end"],
    )
    result = split_longest_interval(df, start="start", end="end", safe_range=5)
    expected = pl.DataFrame(
        {
            "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:05", "2021-01-01 00:00:10"],
            "end": ["2021-01-01 00:00:15", "2021-01-01 00:00:15", "2021-01-01 00:00:15"],
            "value": [2, 2, 2],
        }
    ).with_columns(
        [
            pl.col("start").str.to_datetime(time_unit="ns", time_zone=None),
            pl.col("end").str.to_datetime(time_unit="ns", time_zone=None),
        ]
    )
    assert_frame_equal(result, expected)


def test_filter_datetime_range():
    df = create_dataframe(
        data={
            "start": ["2021-01-01 00:00:00", "2021-01-01 00:00:10"],
            "end": ["2021-01-01 00:00:05", "2021-01-01 00:00:15"],
            "value": [1, 2],
        },
        date_columns=["start", "end"],
    )
    dtime_min = datetime(2021, 1, 1, 0, 0, 0)
    dtime_max = datetime(2021, 1, 1, 0, 0, 10)
    result = filter_datetime_range(df, start="start", end="end", dtime_min=dtime_min, dtime_max=dtime_max)
    expected = pl.DataFrame(
        {"start": ["2021-01-01 00:00:00"], "end": ["2021-01-01 00:00:05"], "value": [1]}
    ).with_columns(
        [
            pl.col("start").str.to_datetime(time_unit="ns", time_zone=None),
            pl.col("end").str.to_datetime(time_unit="ns", time_zone=None),
        ]
    )
    assert_frame_equal(result, expected)


def test_filter_group_length():
    df = pl.DataFrame({"group": ["A", "A", "B", "B", "B", "C"], "value": [1, 2, 3, 2, 1, 3]})
    result = filter_group_length(df, groupby_col="group", min_length=2).sort("group")
    expected = pl.DataFrame({"group": ["A", "B"]})
    assert_frame_equal(result, expected)


def test_generate_windows():
    df = create_dataframe(
        data={"date_time": ["2022-01-01 00:00:00", "2022-01-01 00:00:10", "2022-01-01 00:00:20"], "value": [1, 2, 3]},
        date_columns=["date_time"],
    ).with_row_index("index")
    inds, arr_windows = generate_windows(df, column="date_time", delta_thr=10, sequence_length=2)
    expected_inds = [0, 1, 2]
    expected_arr_windows = np.array([[0, 1], [1, 2]])
    assert inds == expected_inds
    assert np.array_equal(arr_windows, expected_arr_windows)


def test_polars_sort_matches_pandas_sort_for_datetime_and_message():
    dt = [
        datetime(2023, 1, 1, 0, 0, 2),
        datetime(2023, 1, 1, 0, 0, 1),
        datetime(2023, 1, 1, 0, 0, 1),
        datetime(2023, 1, 1, 0, 0, 3),
    ]
    message = ["b", "z", "a", "c"]
    value = [10, 20, 30, 40]

    # External consumer sorts a list of dicts with:
    #   buf.sort(key=itemgetter('@timestamp', 'message'))
    buf = [{"@timestamp": t, "message": m, "value": v} for t, m, v in zip(dt, message, value)]
    buf.sort(key=lambda x: (x["@timestamp"], x["message"]))

    pl_df = pl.DataFrame({"dt": dt, "message": message, "value": value})
    pl_sorted = pl_df.sort("dt", "message", maintain_order=True)

    assert pl_sorted.select(["dt", "message"]).to_dict(as_series=False) == {
        "dt": [x["@timestamp"] for x in buf],
        "message": [x["message"] for x in buf],
    }
