#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module offers a comprehensive suite for processing Fast Healthcare Interoperability Resources
(FHIR) data. It includes the FHIRDataProcessor class along with utility functions designed to
perform various data manipulation tasks. These tasks include normalizing data, filtering outliers
based on predefined criteria, and calculating aggregated metrics such as daily totals, averages,
and moving averages for health metrics identified by LOINC codes. The module is designed to
simplify the manipulation and analysis of healthcare data for research and clinical applications,
working specifically with the structured format provided by the FHIRDataFrame class.

Classes:
    - FHIRDataProcessor: Central class for processing FHIR data, encapsulating methods for
      normalization, outlier filtering, and data aggregation.

Functions:
    - calculate_daily_data: Aggregates daily totals for specific health metrics.
    - calculate_average_data: Calculates daily averages for specific health metrics.
    - calculate_moving_average: Computes moving averages to smooth data series and highlight trends.
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


def _finalize_group(
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

    # Merge the aggregated non-numeric data with the numeric aggregation
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
    group_fhir_dataframe: FHIRDataFrame,
) -> FHIRDataFrame:
    """
    Aggregates daily data for a specific health metric, summing up values within a day.

    Parameters:
        group_fhir_dataframe (FHIRDataFrame): A group of FHIR data entries to be aggregated.

    Returns:
        FHIRDataFrame: Aggregated FHIR data with daily totals for the specified metric,
                       or None if validation fails.
    """
    if group_fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation' for outlier filtering,"
            f"got '{group_fhir_dataframe.resource_type}'."
        )

    if not group_fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    group_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        group_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    aggregated_df = group_fhir_dataframe.df.groupby(
        [
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.LOINC_CODE.value,
        ],
        as_index=False,
    )[ColumnNames.QUANTITY_VALUE.value].sum()

    return FHIRDataFrame(
        data=_finalize_group(group_fhir_dataframe.df, aggregated_df, "Total daily"),
        resource_type=FHIRResourceType.OBSERVATION,
    )


def calculate_average_data(  # pylint: disable=unused-variable
    group_fhir_dataframe: FHIRDataFrame,
) -> FHIRDataFrame:
    """
    Calculates the daily average for a specific health metric across a given time span.

    Parameters:
        group_fhir_dataframe (FHIRDataFrame): A group of FHIR data entries for averaging.

    Returns:
        FHIRDataFrame: Aggregated FHIR data with daily averages for the specified metric.
    """
    if group_fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation' for outlier filtering,"
            f"got '{group_fhir_dataframe.resource_type}'."
        )

    if not group_fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    group_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        group_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    aggregated_df = group_fhir_dataframe.df.groupby(
        [
            ColumnNames.USER_ID.value,
            ColumnNames.EFFECTIVE_DATE_TIME.value,
            ColumnNames.LOINC_CODE.value,
        ],
        as_index=False,
    )[ColumnNames.QUANTITY_VALUE.value].mean()

    return FHIRDataFrame(
        _finalize_group(
            group_fhir_dataframe.df, np.round(aggregated_df, 0), "Daily average"
        ),
        FHIRResourceType.OBSERVATION,
    )


def calculate_moving_average(  # pylint: disable=unused-variable
    flattened_fhir_dataframe: FHIRDataFrame, n=7
) -> FHIRDataFrame:
    """
    Calculates a moving average of the QUANTITY_VALUE over a specified number of days (n)
    for each unique combination of USER_ID and LOINC_CODE. This method is useful
    for smoothing out data series and identifying long-term trends.

    Parameters:
        flattened_fhir_dataframe (FHIRDataFrame): A DataFrame containing flattened FHIR data
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
    if flattened_fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
        raise ValueError(
            f"Resource type must be 'Observation' for outlier filtering,"
            f"got '{flattened_fhir_dataframe.resource_type}'."
        )

    if not flattened_fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    moving_avg_df = flattened_fhir_dataframe.df.groupby(
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
        flattened_fhir_dataframe.df,
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
