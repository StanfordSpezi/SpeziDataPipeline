#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides a collection of functions designed for the processing and analysis of
healthcare data represented in the FHIR (Fast Healthcare Interoperability Resources) format.
It includes capabilities for aggregating data by day, calculating averages, and applying
moving averages to smooth out time-series data. These functions facilitate the examination
of trends and patterns in health metrics over time, making it easier for healthcare professionals,
researchers, and data analysts to derive insights from complex datasets.

The functions are tailored to work with `FHIRDataFrame`, a custom data structure that encapsulates
FHIR data in a pandas DataFrame, allowing for efficient manipulation and analysis of structured
healthcare data. This module emphasizes the flexibility in data analysis tasks, ranging from simple
aggregations to more sophisticated time-series analysis techniques such as moving averages.

Key Features:
- `finalize_group`: Merges aggregated data with non-numeric attributes and applies prefixes to
  descriptive columns, enhancing the readability and interpretability of the results.
- `calculate_daily_data`: Aggregates data on a daily basis, summing up values to provide daily
  totals for specified health metrics, aiding in the analysis of daily trends and variations.
- `calculate_average_data`: Computes daily averages for health metrics, offering insights into
  the typical values and fluctuations within each day.
- `calculate_moving_average`: Applies a moving average to the data, smoothing out short-term
  fluctuations and highlighting long-term trends in health metrics over specified periods.

Use Cases:
- Healthcare data analysts exploring daily patterns, trends, and anomalies in patient health
  metrics.
- Research teams conducting epidemiological studies requiring the aggregation and smoothing of
  time-series data to identify trends and correlations.
- Developers and data scientists building health analytics platforms that require preprocessing,
  normalization, and analysis of healthcare data from FHIR resources.

Example Usage:
The module's functions can be seamlessly integrated into data processing pipelines to enrich
FHIR datasets with aggregated metrics, smoothed time-series, and other derived data forms,
facilitating a wide range of analytical tasks.
from healthcare_data_analysis import calculate_daily_data, calculate_average_data,
    calculate_moving_average

# Assume `fhir_dataframe` is a FHIRDataFrame instance containing patient observation data
daily_totals = calculate_daily_data(fhir_dataframe)
daily_averages = calculate_average_data(fhir_dataframe)
seven_day_moving_average = calculate_moving_average(fhir_dataframe, n=7)

"""

# Related third-party imports
import pandas as pd
import numpy as np

# Local application/library specific imports
from data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
)


def finalize_group(
    original_df: pd.DataFrame, aggregated_df: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    """
    Merges aggregated numeric data with non-numeric data, applying a descriptive prefix
    to the quantity name.

    Parameters:
        original_df (pd.DataFrame): The original DataFrame before aggregation.
        aggregated_df (pd.DataFrame): The DataFrame containing aggregated numeric data.
        prefix (str): A descriptive prefix to add to the QUANTITY_NAME column.

    Returns:
        pd.DataFrame: The final aggregated DataFrame with updated QUANTITY_NAME.
    """
    # Aggregate non-numeric fields by taking the first value in each group
    non_numeric_aggregation = original_df.groupby(
        [
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.LOINC_CODE.value,
        ],
        as_index=False,
    ).agg(
        {
            ColumnNames.APPLE_HEALTH_KIT_CODE.value: "first",
            ColumnNames.QUANTITY_UNIT.value: "first",
            ColumnNames.QUANTITY_NAME.value: "first",
            ColumnNames.DISPLAY.value: "first",
        }
    )

    final_df = pd.merge(
        aggregated_df,
        non_numeric_aggregation,
        on=[
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.LOINC_CODE.value,
        ],
    )

    # Update 'QuantityName' with the prefix
    final_df[ColumnNames.QUANTITY_NAME.value] = final_df[
        ColumnNames.QUANTITY_NAME.value
    ].apply(lambda x: f"{prefix} {x}")

    return final_df


def calculate_daily_data(  # pylint: disable=unused-variable
    fhir_dataframe: FHIRDataFrame,
) -> FHIRDataFrame:
    """
    Aggregates daily data for a specific health metric, summing up values within a day.

    Parameters:
        fhir_dataframe (FHIRDataFrame): A group of FHIR data entries to be aggregated.

    Returns:
        FHIRDataFrame: Aggregated FHIR data with daily totals for the specified metric,
                       or None if validation fails.
    """
    if fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation' for outlier filtering,"
            f"got '{fhir_dataframe.resource_type}'."
        )

    if not fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    aggregated_df = fhir_dataframe.df.groupby(
        [
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.LOINC_CODE.value,
        ],
        as_index=False,
    )[ColumnNames.QUANTITY_VALUE.value].sum()

    return FHIRDataFrame(
        data=finalize_group(fhir_dataframe.df, aggregated_df, "Total daily"),
        resource_type=FHIRResourceType.OBSERVATION,
    )


def calculate_average_data(  # pylint: disable=unused-variable
    fhir_dataframe: FHIRDataFrame,
) -> FHIRDataFrame:
    """
    Calculates the daily average for a specific health metric across a given time span.

    Parameters:
        fhir_dataframe (FHIRDataFrame): A group of FHIR data entries for averaging.

    Returns:
        FHIRDataFrame: Aggregated FHIR data with daily averages for the specified metric.
    """
    if fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation' for outlier filtering,"
            f"got '{fhir_dataframe.resource_type}'."
        )

    if not fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    aggregated_df = fhir_dataframe.df.groupby(
        [
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.LOINC_CODE.value,
        ],
        as_index=False,
    )[ColumnNames.QUANTITY_VALUE.value].mean()

    return FHIRDataFrame(
        finalize_group(fhir_dataframe.df, np.round(aggregated_df, 0), "Daily average"),
        FHIRResourceType.OBSERVATION,
    )


def calculate_moving_average(  # pylint: disable=unused-variable
    fhir_dataframe: FHIRDataFrame, n=7
) -> FHIRDataFrame:
    """
    Calculates a moving average of the QUANTITY_VALUE over a specified number of days (n)
    for each unique combination of USER_ID and LOINC_CODE. This method is useful
    for smoothing out data series and identifying long-term trends.

    Parameters:
        fhir_dataframe (FHIRDataFrame): A DataFrame containing flattened FHIR data
                                        with columns for USER_ID, EFFECTIVE_DATE_TIME,
                                        LOINC_CODE, and QUANTITY_VALUE.
        n (int, optional): The window size in days over which the moving average is
                        calculated. Defaults to 7 days.

    Returns:
        FHIRDataFrame: A DataFrame identical to the input but with 'QuantityValue' replaced
                    by its n-day moving average. All other columns are preserved as is.

    Note:
        This method assumes that the input DataFrame's EFFECTIVE_DATE_TIME column is already
        normalized to date-only values. If EFFECTIVE_DATE_TIME includes time components,
        they should be removed or normalized beforehand to ensure accurate calculations.
    """
    if fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation' for outlier filtering,"
            f"got '{fhir_dataframe.resource_type}'."
        )

    if not fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    moving_avg_df = fhir_dataframe.df.groupby(
        [ColumnNames.USER_ID.value, ColumnNames.LOINC_CODE.value]
    ).apply(
        lambda x: x.sort_values(ColumnNames.EFFECTIVE_DATE_TIME.value)
        .rolling(window=n, on=ColumnNames.EFFECTIVE_DATE_TIME.value)[
            ColumnNames.QUANTITY_VALUE.value
        ]
        .mean()
        .reset_index(drop=True)
    )

    moving_avg_df = moving_avg_df.reset_index()
    result_df = pd.merge(
        fhir_dataframe.df,
        moving_avg_df,
        on=[
            ColumnNames.USER_ID.value,
            ColumnNames.LOINC_CODE.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
        ],
        suffixes=("", "_moving_avg"),
    )
    result_df.rename(
        columns={"QuantityValue_moving_avg": ColumnNames.QUANTITY_VALUE.value},
        inplace=True,
    )

    return FHIRDataFrame(result_df, FHIRResourceType.OBSERVATION)
