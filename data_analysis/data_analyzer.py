#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Related third-party imports
from typing import Any
import pandas as pd
import numpy as np

# Local application/library specific imports
from data_flattening.FHIR_data_flattener import FHIRDataFrame


class FHIRDataProcessor:

    def __init__(self):
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

    def process_fhir_data(self, flattened_fhir_df: FHIRDataFrame) -> FHIRDataFrame:
        self.validate_columns(flattened_fhir_df)
        flattened_fhir_df = flattened_fhir_df.df

        # Normalize 'EffectiveDateTime' to date only
        flattened_fhir_df["EffectiveDateTime"] = pd.to_datetime(
            flattened_fhir_df["EffectiveDateTime"]
        ).dt.date

        processed_fhir_df = pd.DataFrame()

        for (
            user_id,
            effective_dat_time,
            loinc_code,
        ), group_df in flattened_fhir_df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"]
        ):
            # Filter outliers for the group based on LOINC code-specific ranges
            group_FHIRDataFrame = FHIRDataFrame(
                group_df, flattened_fhir_df.resource_type
            )
            filtered_group_FHIRDataFrame = self.filter_outliers(
                group_FHIRDataFrame, self.default_value_ranges.get(loinc_code)
            )

            # Determine the processing function based on the LOINC code
            process_function = self.code_to_function.get(loinc_code)
            if process_function and filtered_group_FHIRDataFrame.df is not None:
                # Apply the processing function to the filtered group
                processed_group_FHIRDataFrame = process_function(
                    filtered_group_FHIRDataFrame
                )
                processed_fhir_df = pd.concat(
                    [processed_fhir_df, processed_group_FHIRDataFrame.df],
                    ignore_index=True,
                )

                processed_FHIRDataFrame = FHIRDataFrame(
                    processed_fhir_df, flattened_fhir_df.resource_type
                )

        return processed_FHIRDataFrame

    def calculate_daily_data(self, group_FHIRDataFrame: FHIRDataFrame) -> FHIRDataFrame:
        self.validate_columns(group_FHIRDataFrame)
        aggregated_df = group_FHIRDataFrame.df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"], as_index=False
        )["QuantityValue"].sum()

        return FHIRDataFrame(
            self._finalize_group(group_FHIRDataFrame.df, aggregated_df, "Total daily"),
            group_FHIRDataFrame.resource_type,
        )

    def calculate_average_data(
        self, group_FHIRDataFrame: FHIRDataFrame
    ) -> FHIRDataFrame:
        self.validate_columns(group_FHIRDataFrame)
        aggregated_df = group_FHIRDataFrame.df.groupby(
            ["UserId", "EffectiveDateTime", "LoincCode"], as_index=False
        )["QuantityValue"].mean()

        return FHIRDataFrame(
            self._finalize_group(
                group_FHIRDataFrame.df, np.round(aggregated_df, 0), "Daily average"
            ),
            group_FHIRDataFrame.resource_type,
        )

    def _finalize_group(
        self, original_df: pd.DataFrame, aggregated_df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
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

    def filter_outliers(
        self: Self @ FHIRDataProcessor,
        flattened_fhir_df: FHIRDataFrame,
        value_range: Any | None = None,
    ) -> FHIRDataFrame:
        self.validate_columns(flattened_fhir_df)

        if value_range is None:
            loinc_code = flattened_fhir_df.df["LoincCode"].iloc[
                0
            ]  # Assumes uniform LoincCode within the DataFrame
            value_range = self.default_value_ranges.get(loinc_code)
            if value_range is None:
                raise ValueError(
                    f"Value range must be defined for 'LoincCode': {loinc_code}."
                )

        lower_bound, upper_bound = value_range
        filtered_df = flattened_fhir_df.df[
            (flattened_fhir_df.df["QuantityValue"] >= lower_bound)
            & (flattened_fhir_df.df["QuantityValue"] <= upper_bound)
        ]
        return FHIRDataFrame(filtered_df, flattened_fhir_df.resource_type)

    def validate_columns(self: Any, flattened_fhir_df: FHIRDataFrame) -> None:

        if flattened_fhir_df.resource_type == "Observation":
            REQUIRED_COLUMNS = [
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
            col for col in REQUIRED_COLUMNS if col not in flattened_fhir_df.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The DataFrame is missing required columns: {missing_columns}"
            )

    def select_data_by_user(
        self: Self @ FHIRDataProcessor, flattened_fhir_df: FHIRDataFrame, user_id: str
    ) -> FHIRDataFrame:
        self.validate_columns(flattened_fhir_df)

        user_df = flattened_fhir_df.df[flattened_fhir_df.df["UserId"] == user_id]
        return FHIRDataFrame(
            user_df.reset_index(drop=True), flattened_fhir_df.resource_type
        )

    def select_data_by_dates(
        self, flattened_fhir_df: FHIRDataFrame, start_date: str, end_date: str
    ) -> FHIRDataFrame:
        """Selects data within a specific date range within a DataFrame."""
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
        self: Any, flattened_fhir_df: FHIRDataFrame, n=7
    ) -> FHIRDataFrame:
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
