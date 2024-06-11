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
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
)

STEP_COUNT_LOINC_CODE = "55423-8"
APPLE_HEALTH_KIT_STEP_COUNT = "HKQuantityTypeIdentifierStepCount"
DISPLAY_STEP_COUNT = "Number of steps in unspecified time Pedometer"
QUANTITY_UNIT_STEPS = "steps"


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

    # Add a ResourceId column to pass validation requirements
    final_df[ColumnNames.RESOURCE_ID.value] = "N/A"

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

    try:
        fhir_dataframe.validate_columns()
    except ValueError as e:
        print(f"Validation failed: {str(e)}")
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

    try:
        fhir_dataframe.validate_columns()
    except ValueError as e:
        print(f"Validation failed: {str(e)}")
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


def calculate_activity_index(  # pylint: disable=unused-variable
    fhir_dataframe: FHIRDataFrame, n: int = 7
) -> FHIRDataFrame:
    """
    Calculate the n-day moving average of step counts for each user in the given FHIRDataFrame,
    ensuring that non-numeric values are propagated correctly across all rows.

    Args:
    fhir_dataframe (FHIRDataFrame): A FHIRDataFrame containing step count observation data.
    n (int): The number of days over which to calculate the moving average.

    Returns:
    FHIRDataFrame: A new FHIRDataFrame containing the updated data.

    Raises:
    ValueError: If the resource type is not 'Observation' or required columns are missing.
    """
    if fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation', got '{fhir_dataframe.resource_type}'."
        )

    try:
        fhir_dataframe.validate_columns()
    except ValueError as e:
        print(f"Validation failed: {str(e)}")
        return None

    fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    )

    if not (
        fhir_dataframe.df[ColumnNames.LOINC_CODE.value] == STEP_COUNT_LOINC_CODE
    ).all():
        print("The function receives as input only step count data.")
        return None

    if fhir_dataframe.df.duplicated(
        subset=[ColumnNames.USER_ID.value, ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).any():
        print(
            "Duplicate entries detected for the same UserId and EffectiveDateTime. "
            "Use process_fhir_data() function before calling calculate_activity_index."
        )
        return None

    fhir_dataframe.df.sort_values(
        by=[ColumnNames.USER_ID.value, ColumnNames.EFFECTIVE_DATE_TIME.value],
        inplace=True,
    )

    def calculate_moving_average(group, n):
        # Ensure every day is accounted for in the range
        group.set_index(ColumnNames.EFFECTIVE_DATE_TIME.value, inplace=True)
        # Resample to daily data, explicitly setting numeric_only=True
        daily = group.resample(
            "D"
        ).asfreq()  # Change from mean() to asfreq() to retain original values
        # Calculate n-day moving average
        daily["MovingAverage"] = (
            daily[ColumnNames.QUANTITY_VALUE.value]
            .rolling(window=n, min_periods=1)
            .mean()
        )
        # Reset index to move EffectiveDateTime back to a column
        daily.reset_index(inplace=True)
        # Include UserId in the output
        daily[ColumnNames.USER_ID.value] = group.name
        # Set constant values
        daily[ColumnNames.LOINC_CODE.value] = STEP_COUNT_LOINC_CODE
        daily[ColumnNames.APPLE_HEALTH_KIT_CODE.value] = APPLE_HEALTH_KIT_STEP_COUNT
        daily[ColumnNames.QUANTITY_UNIT.value] = QUANTITY_UNIT_STEPS
        daily[ColumnNames.DISPLAY.value] = DISPLAY_STEP_COUNT
        daily[ColumnNames.QUANTITY_NAME.value] = f"{n}-day moving average Step Count"
        return daily

    result = fhir_dataframe.df.groupby(ColumnNames.USER_ID.value).apply(
        lambda group: calculate_moving_average(group, n)
    )
    result[ColumnNames.QUANTITY_VALUE.value] = result["MovingAverage"]
    result.reset_index(level=0, drop=True, inplace=True)

    return FHIRDataFrame(result, fhir_dataframe.resource_type)
