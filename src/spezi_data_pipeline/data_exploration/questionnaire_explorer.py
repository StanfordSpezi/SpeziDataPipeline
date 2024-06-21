#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides tools for visualizing risk score derived from FHIR questionnaire responses.
"""

# Standard library imports
from datetime import datetime

# Related third-party imports
import matplotlib.pyplot as plt

# Local application/library specific imports
from spezi_data_pipeline.data_processing.data_processor import (
    select_data_by_dates,
)
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    ColumnNames,
)


class QuestionnaireExplorer:  # pylint: disable=unused-variable
    """
    Provides functionalities to visualize FHIR data. Supports setting up visualization
    parameters such as date range, user IDs for filtering, Y-axis bounds, and the option
    to combine or separate plots for multiple users.

    Attributes:
        start_date (str, optional): Start date for filtering the data for visualization.
            Defaults to None.
        end_date (str, optional): End date for filtering the data for visualization.
            Defaults to None.
        user_ids (list[str], optional): List of user IDs to filter the data for visualization.
            Defaults to None.
        y_lower (float): Lower bound for the Y-axis. Defaults to None.
        y_upper (float): Upper bound for the Y-axis. Defaults to None.
        combine_plots (bool): If True, combines data from multiple users into a single plot.
            Defaults to True.
    """

    def __init__(self):
        """Initializes the QuestionnaireExplorer with default parameters for data visualization."""
        self.start_date = None
        self.end_date = None
        self.user_ids = None
        self.y_lower = None
        self.y_upper = None
        self.combine_plots = True

    def set_date_range(self, start_date: str, end_date: str):
        """
        Sets the start and end dates for filtering the FHIR data before visualization.

        Parameters:
            start_date (str): Start date for data filtering.
            end_date (str): End date for data filtering.
        """
        self.start_date = (
            datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        )
        self.end_date = (
            datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
        )

    def set_user_ids(self, user_ids: list[str]):
        """
        Sets the list of user IDs to filter the FHIR data for visualization.

        Parameters:
            user_ids (list[str]): List of user IDs for data filtering.
        """
        self.user_ids = user_ids

    def set_y_bounds(self, y_lower: float, y_upper: float):
        """
        Sets the lower and upper bounds for the Y-axis of the plot.

        Parameters:
            y_lower (float): Lower bound for the Y-axis.
            y_upper (float): Upper bound for the Y-axis.
        """
        self.y_lower = y_lower
        self.y_upper = y_upper

    def set_combine_plots(self, combine_plots: bool):
        """
        Determines whether to combine plots from multiple users into a single plot
        or create separate plots for each.

        Parameters:
            combine_plots (bool): Flag to combine or separate plots for multiple
                users.
        """
        self.combine_plots = combine_plots

    def create_static_plot(self, fhir_dataframe: FHIRDataFrame) -> list:
        """
        Generates static plots based on the filtered FHIR data, offering combined or
        individual plots for specified users and LOINC codes.

        Parameters:
            fhir_dataframe (FHIRDataFrame): The `FHIRDataFrame` containing the data
                to be visualized.

        Returns:
            list: A list of matplotlib figure objects of the generated plots.
        """
        figures = []

        if self.user_ids is None:
            user_ids = fhir_dataframe[ColumnNames.USER_ID.value].unique()
        else:
            user_ids = (
                self.user_ids if isinstance(self.user_ids, list) else [self.user_ids]
            )

        if self.start_date and self.end_date:
            fhir_dataframe = select_data_by_dates(
                fhir_dataframe, self.start_date, self.end_date
            )

        if fhir_dataframe.empty:
            print("No data for the selected date range.")
            return figures

        if self.combine_plots:
            if fig := self.plot_combined(fhir_dataframe, user_ids):
                figures.append(fig)
        else:
            for user_id in user_ids:
                if fig := self.plot_individual(fhir_dataframe, user_id):
                    figures.append(fig)

        return figures

    def plot_combined(self, df, users_to_plot) -> plt.Figure:
        """
        Generates a combined static plot for multiple users. Each user's data
        is aggregated and plotted within the same figure. This method is useful when
        the class is configured to combine plots from multiple users into a single
        visualization for comparative analysis.

        Parameters:
            df (FHIRDataFrame): A DataFrame containing the data to be plotted.
            users_to_plot (list[str]): A list of user IDs to include in the plot.

        Returns:
            matplotlib.figure.Figure: The figure object representing the combined plot.
        """
        plt.figure(figsize=(10, 6))

        for user_id in users_to_plot:
            user_df = df[df[ColumnNames.USER_ID.value] == user_id]
            plt.plot(
                user_df[ColumnNames.AUTHORED_DATE.value],
                user_df["RiskScore"],
                label=f"User {user_id}",
            )

        date_range_title = (
            "for all dates"
            if not self.start_date and not self.end_date
            else f"from {self.start_date} to {self.end_date}"
        )
        plt.title(f"Risk Score {date_range_title}")
        plt.xlabel("Date")
        plt.ylabel("Risk Score")
        plt.legend()
        plt.xticks(rotation=45)
        if self.y_lower is not None and self.y_upper is not None:
            plt.ylim(self.y_lower, self.y_upper)
        plt.tight_layout()

        fig = plt.gcf()
        plt.show()
        return fig

    def plot_individual(self, df, user_id) -> plt.Figure:
        """
        Generates individual static plots for each specified user. For each user, their
        data is aggregated and plotted in a separate figure. This method is called to
        generate detailed plots for individual users.

        Parameters:
            df (FHIRDataFrame): A `DataFrame` containing the data to be plotted.
            user_id (str): The user ID for which to generate the plot.

        Returns:
            matplotlib.figure.Figure: The figure object representing the individual plot.
            If no data is found for the specified user ID, returns None.
        """
        if user_id is None:
            print("User ID must be provided for individual plots.")
            return None

        user_df = df[df[ColumnNames.USER_ID.value] == user_id]
        if user_df.empty:
            print(f"No data found for user ID {user_id}.")
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(
            user_df[ColumnNames.AUTHORED_DATE.value],
            user_df["RiskScore"],
            label=f"User {user_id}",
        )

        date_range_title = (
            "for all dates"
            if not self.start_date and not self.end_date
            else f"from {self.start_date} to {self.end_date}"
        )
        plt.title(f"Risk Score for User {user_id} {date_range_title}")
        plt.xlabel("Date")
        plt.ylabel("Risk Score")
        plt.legend()
        plt.xticks(rotation=45)
        if self.y_lower is not None and self.y_upper is not None:
            plt.ylim(self.y_lower, self.y_upper)
        plt.tight_layout()

        fig = plt.gcf()
        plt.show()
        return fig
