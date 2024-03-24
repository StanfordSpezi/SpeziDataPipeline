#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
# Standard library imports
from datetime import datetime, date
from typing import Any, List, Optional

# Related third-party imports
import matplotlib.pyplot as plt

# Local application/library specific imports
from data_analysis.data_analyzer import FHIRDataProcessor
from data_flattening.FHIR_data_flattener import FHIRDataFrame


class DataVisualizer(FHIRDataProcessor):
    def __init__(self):
        super().__init__()
        self.start_date = None
        self.end_date = None
        self.user_ids = None
        self.y_lower = 50
        self.y_upper = 1000
        self.same_plot = True
        self.dpi = 300

    def set_date_range(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def set_user_ids(self, user_ids: List[str]):
        self.user_ids = user_ids

    def set_y_bounds(self, y_lower: float, y_upper: float):
        self.y_lower = y_lower
        self.y_upper = y_upper

    def set_same_plot(self, same_plot: bool):
        self.same_plot = same_plot

    def set_dpi(self, dpi: float):
        self.dpi = dpi

    def create_static_plot(
        self: Any, flattened_FHIRDataFrame: FHIRDataFrame
    ) -> Optional[plt.Figure]:
        if not isinstance(
            flattened_FHIRDataFrame.df["EffectiveDateTime"].iloc[0], date
        ):
            print("The date type should be of type date.")
            return

        if flattened_FHIRDataFrame.df["LoincCode"].nunique() != 1:
            print(
                "Error: More than one unique LoincCode found. Each plot should be based on a single LoincCode."
            )
            return

        self.validate_columns(flattened_FHIRDataFrame)
        flattened_df = flattened_FHIRDataFrame.df
        if self.start_date:
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        if self.end_date:
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date()
        if self.start_date and self.end_date:
            flattened_df = flattened_df[
                (flattened_df["EffectiveDateTime"] >= self.start_date)
                & (flattened_df["EffectiveDateTime"] <= self.end_date)
            ]

        users_to_plot = (
            self.user_ids if self.user_ids else flattened_df["UserId"].unique()
        )

        if self.same_plot:
            plt.figure(figsize=(10, 6), dpi=self.dpi)
            for uid in users_to_plot:
                user_df = flattened_df[flattened_df["UserId"] == uid]
                aggregated_data = (
                    user_df.groupby("EffectiveDateTime")["QuantityValue"]
                    .sum()
                    .reset_index()
                )
                plt.bar(
                    aggregated_data["EffectiveDateTime"],
                    aggregated_data["QuantityValue"],
                    edgecolor="black",
                    linewidth=1.5,
                    label=f"User {uid}",
                )
            plt.ylim(self.y_lower, self.y_upper)
            plt.legend()
            plt.title(
                f"{flattened_df['QuantityName'].iloc[0]} from {self.start_date} to {self.end_date}"
            )
            plt.xlabel("Date")
            plt.ylabel(
                f"{flattened_df['QuantityName'].iloc[0]} ({flattened_df['QuantityUnit'].iloc[0]})"
            )
            plt.xticks(rotation=45)
            plt.yticks()
            plt.tight_layout()
            fig = plt.gcf()
            plt.show()

        else:
            for uid in users_to_plot:
                plt.figure(figsize=(10, 6), dpi=self.dpi)
                user_df = flattened_df[flattened_df["UserId"] == uid]
                aggregated_data = (
                    user_df.groupby("EffectiveDateTime")["QuantityValue"]
                    .sum()
                    .reset_index()
                )
                plt.bar(
                    aggregated_data["EffectiveDateTime"],
                    aggregated_data["QuantityValue"],
                    edgecolor="black",
                    linewidth=1.5,
                    label=f"User {uid}",
                )
                plt.legend()
                plt.title(
                    f"{user_df['QuantityName'].iloc[0]} for User {uid} from {self.start_date} to {self.end_date}"
                )
                plt.xlabel("Date")
                plt.ylabel(
                    f"{user_df['QuantityName'].iloc[0]} ({user_df['QuantityUnit'].iloc[0]})"
                )
                plt.xticks(rotation=45)
                plt.yticks()
                plt.tight_layout()
                if len(users_to_plot) == 1:
                    fig = plt.gcf()
                plt.show()

        return fig
