#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module encompasses the functionality necessary for handling various aspects of FHIR data
processing, including but not limited to normalization, outlier detection and filtering, and the
aggregation of healthcare data metrics. It is designed to work with instances of `FHIRDataFrame`,
a custom data structure that represents flattened FHIR data in a tabular format suitable for
analysis.

Key Features:
- `FHIRDataProcessor`: A central class that provides methods for processing FHIR data, leveraging
    predefined code mappings to apply specific processing functions and value ranges for different
    health data metrics.
- `select_data_by_user` and `select_data_by_dates`: Utility functions that facilitate the filtering
    of FHIR data based on user IDs and date ranges, respectively.
- Code mappings and processing functions: Mechanisms for associating LOINC codes with specific
    processing routines and value ranges, enabling targeted and meaningful analysis of healthcare
    metrics.
"""

# Related third-party imports
from typing import Any

# pylint: disable=duplicate-code
import pandas as pd

# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
    ColumnNames,
)

# pylint: enable=duplicate-code
from spezi_data_pipeline.data_processing.code_mapping import CodeProcessor


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

        if flattened_fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
            print(
                f"The FHIRDataFrame contains {flattened_fhir_dataframe.resource_type} data."
                "The data were not processed."
            )
            return flattened_fhir_dataframe

        try:
            flattened_fhir_dataframe.validate_columns()
        except ValueError as e:
            print(f"Validation failed: {str(e)}")
            return None

        flattened_df = flattened_fhir_dataframe.df

        flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value] = pd.to_datetime(
            flattened_df[ColumnNames.EFFECTIVE_DATE_TIME.value]
        ).dt.date

        processed_df = pd.DataFrame()

        # Iterate over each group defined by unique combinations of UserID, EffectiveDateTime,
        # and LOINCCode
        for _, group_df in flattened_df.groupby(
            [
                ColumnNames.USER_ID.value,
                ColumnNames.EFFECTIVE_DATE_TIME.value,
                ColumnNames.LOINC_CODE.value,
            ]
        ):
            group_fhir_dataframe = FHIRDataFrame(
                group_df, flattened_fhir_dataframe.resource_type
            )
            filtered_group_df = self.filter_outliers(
                group_fhir_dataframe,
                self.code_processor.default_value_ranges.get(
                    group_df[ColumnNames.LOINC_CODE.value].iloc[0], None
                ),
            )

            process_function = self.code_processor.code_to_function.get(
                group_df[ColumnNames.LOINC_CODE.value].iloc[0]
            )

            if process_function:
                processed_group_df = process_function(filtered_group_df)
            else:
                processed_group_df = filtered_group_df

            if isinstance(processed_group_df.df, pd.DataFrame):
                processed_df = pd.concat(
                    [processed_df, processed_group_df.df], ignore_index=True
                )

        if processed_df.empty:
            print("No data was processed.")
            return None

        return FHIRDataFrame(
            processed_df, resource_type=flattened_fhir_dataframe.resource_type
        )

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
        if flattened_fhir_dataframe.resource_type != FHIRResourceType.OBSERVATION:
            raise ValueError(
                f"{flattened_fhir_dataframe.resource_type} are not supported for outlier filtering."
            )

        try:
            flattened_fhir_dataframe.validate_columns()
        except ValueError as e:
            print(f"Validation failed: {str(e)}")
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
    try:
        flattened_fhir_dataframe.validate_columns()
    except ValueError as e:
        print(f"Validation failed: {str(e)}")
        return None

    if flattened_fhir_dataframe.resource_type in [
        FHIRResourceType.OBSERVATION,
        FHIRResourceType.ECG_OBSERVATION,
    ]:
        date_column = ColumnNames.EFFECTIVE_DATE_TIME.value
    elif (
        flattened_fhir_dataframe.resource_type
        == FHIRResourceType.QUESTIONNAIRE_RESPONSE
    ):
        date_column = ColumnNames.AUTHORED_DATE.value
    else:
        raise ValueError("Unsupported FHIR resource type for user selection")

    flattened_fhir_dataframe.df[date_column] = pd.to_datetime(
        flattened_fhir_dataframe.df[date_column]
    ).dt.date

    user_df = flattened_fhir_dataframe.df[
        flattened_fhir_dataframe.df[ColumnNames.USER_ID.value] == user_id
    ]

    return FHIRDataFrame(
        user_df.reset_index(drop=True),
        resource_type=flattened_fhir_dataframe.resource_type,
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
    try:
        flattened_fhir_dataframe.validate_columns()
    except ValueError as e:
        print(f"Validation failed: {str(e)}")
        return None

    if flattened_fhir_dataframe.resource_type in [
        FHIRResourceType.OBSERVATION,
        FHIRResourceType.ECG_OBSERVATION,
    ]:
        date_column = ColumnNames.EFFECTIVE_DATE_TIME.value
    elif (
        flattened_fhir_dataframe.resource_type
        == FHIRResourceType.QUESTIONNAIRE_RESPONSE
    ):
        date_column = ColumnNames.AUTHORED_DATE.value
    else:
        raise ValueError("Unsupported FHIR resource type for date selection")

    flattened_fhir_dataframe.df[date_column] = pd.to_datetime(
        flattened_fhir_dataframe.df[date_column]
    ).dt.date

    start_datetime = pd.to_datetime(start_date).date()
    end_datetime = pd.to_datetime(end_date).date()

    filtered_df = flattened_fhir_dataframe.df[
        (flattened_fhir_dataframe.df[date_column] >= start_datetime)
        & (flattened_fhir_dataframe.df[date_column] <= end_datetime)
    ]

    return FHIRDataFrame(
        filtered_df.reset_index(drop=True),
        resource_type=flattened_fhir_dataframe.resource_type,
    )

def extract_latest_user_interaction(  # pylint: disable=unused-variable
    flattend_fhir_df: pd.DataFrame
    )->None:
    """
    Extracts the latest interaction of each user from a flattened fhir DataFrame and saves results to a CSV file. 
    Parameters:
        flattend_fhir_df (pd.DataFrame): The flattened_fhir `DataFrame`.
    """
    #Base case
    if flattend_fhir_df.empty:
        return

    #First convert column to datetime for comparison
    flattend_fhir_df['EffectiveDateTime'] = pd.to_datetime(flattend_fhir_df['EffectiveDateTime'], format='%d.%m.%y')
    #Filter the most recent entry for each userid 
    most_recent_df=flattend_fhir_df.loc[flattend_fhir_df.groupby('UserId')['EffectiveDateTime'].idxmax()]
    most_recent_df=most_recent_df[['UserId','EffectiveDateTime']]#select the relevant cols
    most_recent_df.rename(columns={'EffectiveDateTime': 'LastUserInteraction'}, inplace=True)
    most_recent_df.to_csv('output.csv')
    return