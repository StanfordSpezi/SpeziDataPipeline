#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Standard library imports
from enum import Enum

# Related third-party imports
from typing import Any
import pandas as pd
import numpy as np

# Local application/library specific imports
from data_flattening.fhir_data_flattener import FHIRDataFrame


class FHIRResourceType(Enum):
    """
    Enumeration of FHIR resource types.

    This enum provides a list of FHIR resource types used in the application, ensuring
    consistency and preventing typos in resource type handling.

    Attributes:
        OBSERVATION (str): Represents an observation resource type.

    Note:
        The `.value` attribute is used to access the string value of the enum members.
    """

    OBSERVATION = "Observation"


class FHIRDataProcessor:
    """
    Processes FHIR data for analysis and reporting, focusing on transforming and
    normalizing health data according to specific metrics and value ranges.

    This class maps LOINC codes to specific processing functions and applies outlier
    filtering and data aggregation strategies to compute daily averages or totals
    for health metrics.

    Attributes:
        code_to_function (dict): Maps LOINC codes to processing functions for specific health data
                            metrics.
        default_value_ranges (dict): Specifies default value ranges for outlier filtering based on
                                LOINC codes.
    """

    def __init__(self):
        """
        Initializes the FHIRDataProcessor with mappings from LOINC codes to processing
        functions and default value ranges for various health metrics.
        """
        # Maps LOINC codes to the corresponding processing function
        self.code_to_function = {
            # Averaged per Day LOINC codes
            "9052-2": self.calculate_daily_data,  # Example LOINC codes for demonstration
            "55423-8": self.calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryProtein": self.calculate_average_data,
            # Add other LOINC codes here
        }

        self.default_value_ranges = {
            "55423-8": (0, 30000),  # Number of steps
            "HKQuantityTypeIdentifierDietaryProtein": (0, 500),
            "9052-2": (0, 2700),  # Dietary energy consumed in calories
        }

    def process_fhir_data(
        self: "FHIRDataProcessor", flattened_fhir_df: FHIRDataFrame
    ) -> FHIRDataFrame:
        """
        Processes flattened FHIR data, applying code-specific processing functions
        and filtering outliers based on predefined value ranges.

        Parameters:
            flattened_fhir_df (FHIRDataFrame): A DataFrame containing flattened FHIR data.

        Returns:
            FHIRDataFrame: A DataFrame containing processed FHIR data.
        """
        self.validate_columns(flattened_fhir_df)
        flattened_fhir_df = flattened_fhir_df.df

        # Normalize 'EffectiveDateTime' to date only
        flattened_fhir_df["EffectiveDateTime"] = pd.to_datetime(
            flattened_fhir_df["EffectiveDateTime"]
        ).dt.date

        processed_fhir_df = pd.DataFrame()

        for (
            _,
            _,
            loinc_code,
        ), group_df in flattened_fhir_df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"]
        ):
            # Filter outliers for the group based on LOINC code-specific ranges
            group_fhir_dataframe = FHIRDataFrame(
                group_df, flattened_fhir_df.resource_type
            )
            filtered_group_fhir_dataframe = self.filter_outliers(
                group_fhir_dataframe, self.default_value_ranges.get(loinc_code)
            )

            # Determine the processing function based on the LOINC code
            process_function = self.code_to_function.get(loinc_code)
            if process_function and filtered_group_fhir_dataframe.df is not None:
                # Apply the processing function to the filtered group
                processed_group_fhir_dataframe = process_function(
                    filtered_group_fhir_dataframe
                )
                processed_fhir_df = pd.concat(
                    [processed_fhir_df, processed_group_fhir_dataframe.df],
                    ignore_index=True,
                )

                processed_FHIRDataFrame = FHIRDataFrame(
                    processed_fhir_df, flattened_fhir_df.resource_type
                )

        return processed_FHIRDataFrame

    def calculate_daily_data(
        self: "FHIRDataProcessor", group_fhir_dataframe: FHIRDataFrame
    ) -> FHIRDataFrame:
        """
        Aggregates daily data for a specific health metric, summing up values within a day.

        Parameters:
            group_fhir_dataframe (FHIRDataFrame): A group of FHIR data entries to be aggregated.

        Returns:
            FHIRDataFrame: Aggregated FHIR data with daily totals for the specified metric.
        """
        self.validate_columns(group_fhir_dataframe)
        aggregated_df = group_fhir_dataframe.df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"], as_index=False
        )["QuantityValue"].sum()

        return FHIRDataFrame(
            _finalize_group(group_fhir_dataframe.df, aggregated_df, "Total daily"),
            group_fhir_dataframe.resource_type,
        )

    def calculate_average_data(
        self: "FHIRDataProcessor", group_fhir_dataframe: FHIRDataFrame
    ) -> FHIRDataFrame:
        """
        Calculates the daily average for a specific health metric across a given time span.

        Parameters:
            group_fhir_dataframe (FHIRDataFrame): A group of FHIR data entries for averaging.

        Returns:
            FHIRDataFrame: Aggregated FHIR data with daily averages for the specified metric.
        """
        self.validate_columns(group_fhir_dataframe)
        aggregated_df = group_fhir_dataframe.df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"], as_index=False
        )["QuantityValue"].mean()

        return FHIRDataFrame(
            _finalize_group(
                group_fhir_dataframe.df, np.round(aggregated_df, 0), "Daily average"
            ),
            group_fhir_dataframe.resource_type,
        )

    def filter_outliers(
        self: "FHIRDataProcessor",
        flattened_fhir_df: FHIRDataFrame,
        value_range: Any | None = None,
    ) -> FHIRDataFrame:
        """
        Filters out data points that fall outside the specified value range, effectively
        removing outliers from the dataset.

        Parameters:
            flattened_fhir_df (FHIRDataFrame): The DataFrame to filter.
            value_range (Tuple[int, int], optional): The inclusive range of acceptable values.

        Returns:
            FHIRDataFrame: The filtered DataFrame with outliers removed.
        """
        self.validate_columns(flattened_fhir_df)

        if value_range is None:
            loinc_code = flattened_fhir_df.df["LoincCode"].iloc[
                0
            ]  # Assumes uniform LoincCode within the DataFrame
            if (value_range := self.default_value_ranges.get(loinc_code)) is None:
                raise ValueError(
                    f"Value range must be defined for 'LoincCode': {loinc_code}."
                )

        lower_bound, upper_bound = value_range
        filtered_df = flattened_fhir_df.df[
            (flattened_fhir_df.df["QuantityValue"] >= lower_bound)
            & (flattened_fhir_df.df["QuantityValue"] <= upper_bound)
        ]
        return FHIRDataFrame(filtered_df, flattened_fhir_df.resource_type)

    def select_data_by_user(
        self: "FHIRDataProcessor", flattened_fhir_df: FHIRDataFrame, user_id: str
    ) -> FHIRDataFrame:
        """
        Selects data corresponding to a specific user from a DataFrame.

        Parameters:
            flattened_fhir_df (FHIRDataFrame): The DataFrame from which to select data.
            user_id (str): The user ID for which data is to be selected.

        Returns:
            FHIRDataFrame: A DataFrame containing only data for the specified user.
        """
        self.validate_columns(flattened_fhir_df)

        user_df = flattened_fhir_df.df[flattened_fhir_df.df["UserId"] == user_id]
        return FHIRDataFrame(
            user_df.reset_index(drop=True), flattened_fhir_df.resource_type
        )

    def select_data_by_dates(
        self: "FHIRDataProcessor",
        flattened_fhir_df: FHIRDataFrame,
        start_date: str,
        end_date: str,
    ) -> FHIRDataFrame:
        """
        Selects data within a specific date range from a DataFrame.

        Parameters:
            flattened_fhir_df (FHIRDataFrame): The DataFrame from which to select data.
            start_date (str): The start date of the range (inclusive).
            end_date (str): The end date of the range (inclusive).

        Returns:
            FHIRDataFrame: A DataFrame containing data within the specified date range.
        """
        self.validate_columns(flattened_fhir_df)

        start_datetime = pd.to_datetime(start_date).tz_localize(None)
        end_datetime = pd.to_datetime(end_date).tz_localize(None)

        flattened_fhir_df.df["EffectiveDateTime"] = pd.to_datetime(
            flattened_fhir_df.df["EffectiveDateTime"]
        ).dt.tz_localize(None)

        filtered_df = flattened_fhir_df.df[
            (flattened_fhir_df.df["EffectiveDateTime"] >= start_datetime)
            & (flattened_fhir_df.df["EffectiveDateTime"] <= end_datetime)
        ]

        return FHIRDataFrame(
            filtered_df.reset_index(drop=True), flattened_fhir_df.resource_type
        )

    def calculate_moving_average(
        self: "FHIRDataProcessor", flattened_fhir_df: FHIRDataFrame, n=7
    ) -> FHIRDataFrame:
        """
        Calculates a moving average of the 'QuantityValue' over a specified number of days (n)
        for each unique combination of 'UserId' and 'LoincCode'. This method is useful
        for smoothing out data series and identifying long-term trends.

        Parameters:
            flattened_fhir_df (FHIRDataFrame): A DataFrame containing flattened FHIR data
                                            with columns for 'UserId', 'EffectiveDateTime',
                                            'LoincCode', and 'QuantityValue'.
            n (int, optional): The window size in days over which the moving average is
                            calculated. Defaults to 7 days.

        Returns:
            FHIRDataFrame: A DataFrame identical to the input but with 'QuantityValue' replaced
                        by its n-day moving average. All other columns are preserved as is.

        Note:
            This method assumes that the input DataFrame's 'EffectiveDateTime' column is already
            normalized to date-only values. If 'EffectiveDateTime' includes time components,
            they should be removed or normalized beforehand to ensure accurate calculations.
        """
        self.validate_columns(flattened_fhir_df)

        flattened_fhir_df.df["EffectiveDateTime"] = pd.to_datetime(
            flattened_fhir_df.df["EffectiveDateTime"]
        ).dt.date

        moving_avg_df = flattened_fhir_df.df.groupby(["UserId", "LoincCode"]).apply(
            lambda x: x.sort_values("EffectiveDateTime")
            .rolling(window=n, on="EffectiveDateTime")["QuantityValue"]
            .mean()
            .reset_index(drop=True)
        )

        moving_avg_df = moving_avg_df.reset_index()
        result_df = pd.merge(
            flattened_fhir_df.df,
            moving_avg_df,
            on=["UserId", "LoincCode", "EffectiveDateTime"],
            suffixes=("", "_moving_avg"),
        )
        result_df.rename(
            columns={"QuantityValue_moving_avg": "QuantityValue"}, inplace=True
        )

        return FHIRDataFrame(result_df, flattened_fhir_df.resource_type)


def validate_columns(flattened_fhir_df: FHIRDataFrame) -> None:
    """
    Validates that the DataFrame contains all required columns for processing.
    Raises a ValueError if any required column is missing.

    Parameters:
        flattened_fhir_df (FHIRDataFrame): The DataFrame to validate.

    Raises:
        ValueError: If any required columns are missing.
    """

    if flattened_fhir_df.resource_type == FHIRResourceType.OBSERVATION.value:
        required_columns = [
            "UserId",
            "EffectiveDateTime",
            "QuantityName",
            "QuantityUnit",
            "QuantityValue",
            "LoincCode",
            "Display",
            "AppleHealthKitCode",
        ]

    missing_columns = [
        col for col in required_columns if col not in flattened_fhir_df.df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"The DataFrame is missing required columns: {missing_columns}"
        )

def _finalize_group(
        original_df: pd.DataFrame,
        aggregated_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """
        Merges aggregated numeric data with non-numeric data, applying a descriptive prefix
        to the quantity name.

        Parameters:
            original_df (pd.DataFrame): The original DataFrame before aggregation.
            aggregated_df (pd.DataFrame): The DataFrame containing aggregated numeric data.
            prefix (str): A descriptive prefix to add to the 'QuantityName' column.

        Returns:
            pd.DataFrame: The final aggregated DataFrame with updated 'QuantityName'.
        """
        # Aggregate non-numeric fields by taking the first value in each group
        non_numeric_aggregation = original_df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"], as_index=False
        ).agg(
            {
                "AppleHealthKitCode": "first",
                "QuantityUnit": "first",
                "QuantityName": "first",
                "Display": "first",
            }
        )

        # Merge the aggregated non-numeric data with the numeric aggregation
        final_df = pd.merge(
            aggregated_df,
            non_numeric_aggregation,
            on=["UserId", "EffectiveDateTime", "LoincCode"],
        )

        # Update 'QuantityName' with the prefix
        final_df["QuantityName"] = final_df["QuantityName"].apply(
            lambda x: f"{prefix} {x}"
        )

        return final_df
