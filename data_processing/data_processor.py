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
simplify the manipulation and analysis of healthcare data for research and clinical
applications, working specifically with the structured format provided by the FHIRDataFrame class.

Classes:
    - FHIRDataProcessor: Central class for processing FHIR data, encapsulating methods for
      normalization, outlier filtering, and data aggregation.

Functions:
    - calculate_daily_data: Aggregates daily totals for specific health metrics.
    - calculate_average_data: Calculates daily averages for specific health metrics.
    - calculate_moving_average: Computes moving averages to smooth data series and highlight trends.
"""


# Related third-party imports
from typing import Any
import pandas as pd

# Local application/library specific imports
from data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
)
from .code_mapping import CodeProcessor


class FHIRDataProcessor:  # pylint: disable=unused-variable
    """
    Provides functionalities for processing and analyzing FHIR data, tailored to work
    with FHIRDataFrame instances. It supports operations such as normalization, outlier
    filtering, and aggregation of healthcare data metrics based on LOINC codes.

    The processor leverages code mappings to apply specific processing functions for
    different health data metrics, facilitating the computation of daily totals and averages,
    and ensuring the data falls within acceptable value ranges for meaningful analysis.

    Attributes:
        code_processor (CodeProcessor): An instance of CodeProcessor containing mappings from
                                         LOINC codes to processing functions and value ranges.
    """

    def __init__(self):
        """
        Initializes the FHIRDataProcessor with a CodeProcessor instance to manage LOINC code
        mappings to processing functions and value ranges.
        """
        self.code_processor = CodeProcessor()

    def process_fhir_data(
        self, flattened_fhir_dataframe: FHIRDataFrame
    ) -> FHIRDataFrame:
        """
        Processes a given FHIRDataFrame by applying normalization, filtering outliers based on
        predefined value ranges, and computing aggregated metrics such as daily totals and averages.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): A FHIRDataFrame containing flattened FHIR
                                                      data.

        Returns:
            FHIRDataFrame: A processed FHIRDataFrame with applied normalization, outlier filtering,
                           and aggregated metrics computations.

        Raises:
            ValueError: If the input data frame is not a valid FHIRDataFrame or does not meet
                        the required structure for processing.
        """
        if not isinstance(flattened_fhir_dataframe, FHIRDataFrame):
            print("Please use a valid FHIRDataFrame.")
            return None

        if not flattened_fhir_dataframe.validate_columns():
            return None

        flattened_df = flattened_fhir_dataframe.df

        # Normalize 'EffectiveDateTime' to date only
        flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
            flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value]
        ).dt.date

        processed_df = pd.DataFrame()

        for (
            _,
            _,
            loinc_code,
        ), group_df in flattened_df.groupby(
            [
                ColumnNames.USER_ID.value,
                ColumnNames.EFFECTIVE_DATE_TIME.value,
                ColumnNames.LOINC_CODE.value,
            ]
        ):
            # Filter outliers for the group based on LOINC code-specific ranges
            group_fhir_dataframe = FHIRDataFrame(
                group_df, FHIRResourceType(flattened_fhir_dataframe.resource_type)
            )
            filtered_group_fhir_dataframe = self.filter_outliers(
                group_fhir_dataframe,
                self.code_processor.default_value_ranges.get(loinc_code),
            )

            # Determine the processing function based on the LOINC code
            process_function = self.code_processor.code_to_function.get(loinc_code)
            if process_function and filtered_group_fhir_dataframe.df is not None:
                # Apply the processing function to the filtered group
                processed_group_fhir_dataframe = process_function(
                    filtered_group_fhir_dataframe
                )
                processed_df = pd.concat(
                    [processed_df, processed_group_fhir_dataframe.df],
                    ignore_index=True,
                )

                processed_fhir_dataframe = FHIRDataFrame(
                    processed_df,
                    FHIRResourceType(flattened_fhir_dataframe.resource_type),
                )
            else:
                processed_fhir_dataframe = FHIRDataFrame(
                    flattened_fhir_dataframe.df,
                    FHIRResourceType(flattened_fhir_dataframe.resource_type),
                )

        return processed_fhir_dataframe

    def filter_outliers(
        self,
        flattened_fhir_dataframe: FHIRDataFrame,
        value_range: Any | None = None,
    ) -> FHIRDataFrame:
        """
        Filters outliers from a FHIRDataFrame based on specified value ranges for each LOINC code,
        removing data points that fall outside acceptable value ranges to ensure data quality.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame to be filtered.
            value_range (tuple[int, int] | None): An optional tuple specifying the inclusive range
                                                  of acceptable values. If None, default value
                                                  ranges based on LOINC codes are used.

        Returns:
            FHIRDataFrame: A filtered FHIRDataFrame with outliers removed.

        Raises:
            ValueError: If the resource type of the input data frame is not 'Observation', as
                        outlier filtering is currently supported only for Observation data.
        """
        if (
            flattened_fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION
            and flattened_fhir_dataframe.resource_type
            != FHIRResourceType.ECG_OBSERVATION
        ):
            raise ValueError(
                f"Resource type must be 'Observation' for outlier filtering,"
                f"got '{flattened_fhir_dataframe.resource_type}'."
            )

        if not flattened_fhir_dataframe.validate_columns():
            print("Validation failed: Column requirement is not satisfied.")
            return None

        # Filter data points based on LOINC code specific value ranges
        def filter_by_loinc(row):
            if value_range is not None:
                # If a global value_range is defined, use it
                return (
                    value_range[0]
                    <= row[ColumnNames.QUANTITY_VALUE.value]
                    <= value_range[1]
                )
            # Otherwise, retrieve the value range specific to the LOINC code
            loinc_code_range = self.code_processor.default_value_ranges.get(
                row[ColumnNames.LOINC_CODE.value]
            )
            if loinc_code_range:
                return (
                    loinc_code_range[0]
                    <= row[ColumnNames.QUANTITY_VALUE.value]
                    <= loinc_code_range[1]
                )
            return True

        filtered_df = flattened_fhir_dataframe.df[
            flattened_fhir_dataframe.df.apply(filter_by_loinc, axis=1)
        ]

        return FHIRDataFrame(filtered_df, FHIRResourceType.OBSERVATION)


def select_data_by_user(  # pylint: disable=unused-variable
    flattened_fhir_dataframe: FHIRDataFrame, user_id: str
) -> FHIRDataFrame:
    """
    Selects and returns data for a specific user from a FHIRDataFrame based on the user ID.

    Parameters:
        flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame from which to select data.
        user_id (str): The user ID for which to select data.

    Returns:
        FHIRDataFrame: A new FHIRDataFrame containing only the data for the specified user.

    Raises:
        ValueError: If the input data frame does not meet the required column structure.
    """
    if not flattened_fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    user_df = flattened_fhir_dataframe.df[
        flattened_fhir_dataframe.df[ColumnNames.USER_ID.value] == user_id
    ]
    return FHIRDataFrame(
        user_df.reset_index(drop=True),
        FHIRResourceType(flattened_fhir_dataframe.resource_type),
    )


def select_data_by_dates(  # pylint: disable=unused-variable
    flattened_fhir_dataframe: FHIRDataFrame,
    start_date: str,
    end_date: str,
) -> FHIRDataFrame:
    """
    Filters and returns data from a FHIRDataFrame within a specified date range.

    Parameters:
        flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame from which to filter data.
        start_date (str): The start date of the range (inclusive).
        end_date (str): The end date of the range (inclusive).

    Returns:
        FHIRDataFrame: A new FHIRDataFrame containing only the data within the specified date range.

    Raises:
        ValueError: If the input data frame does not meet the required column structure or if
                    date parameters are not in a valid format.
    """
    if not flattened_fhir_dataframe.validate_columns():
        print("Validation failed: Column requirement is not satisfied.")
        return None

    flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.date

    start_datetime = pd.to_datetime(start_date).tz_localize(None)
    end_datetime = pd.to_datetime(end_date).tz_localize(None)

    flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
        flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
    ).dt.tz_localize(None)

    filtered_df = flattened_fhir_dataframe.df[
        (
            flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
            >= start_datetime
        )
        & (
            flattened_fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
            <= end_datetime
        )
    ]

    return FHIRDataFrame(
        filtered_df.reset_index(drop=True),
        FHIRResourceType(flattened_fhir_dataframe.resource_type),
    )
