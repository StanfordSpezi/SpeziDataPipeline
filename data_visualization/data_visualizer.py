#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
 The DataVisualizer module, building upon the functionalities provided
 by the FHIRDataProcessor, is designed to visualize health-related data
 processed from FHIR (Fast Healthcare Interoperability Resources) formats.
 It supports generating static plots for data analysis and insights, offering a
 range of customization options to cater to different visualization needs.
 
 """

# Related third-party imports
import matplotlib.pyplot as plt

# Local application/library specific imports
from data_processing.data_processor import (
    FHIRDataProcessor,
    select_data_by_dates,
    select_data_by_user,
)
from data_flattening.fhir_resources_flattener import FHIRDataFrame, ColumnNames


class DataVisualizer(FHIRDataProcessor):  # pylint: disable=unused-variable
    """
    Provides functionalities to visualize FHIR data, extending the FHIRDataProcessor class.
    It supports setting up various visualization parameters such as date range, user IDs for
    filtering, Y-axis bounds, and the option to combine multiple users' data into a single plot
    or separate plots for each user.

    Attributes:
        start_date (str, optional): Start date for filtering the data for visualization.
                                    Defaults to None.
        end_date (str, optional): End date for filtering the data for visualization.
                                  Defaults to None.
        user_ids (list[str], optional): List of user IDs to filter the data for visualization.
                                        Defaults to None.
        y_lower (float): Lower bound for the Y-axis. Defaults to 50.
        y_upper (float): Upper bound for the Y-axis. Defaults to 1000.
        combine_plots (bool): If True, combines data from multiple users into a single plot.
                            Otherwise, creates separate plots for each user. Defaults to True.

    Methods:
        set_date_range(start_date, end_date): Sets the date range for data filtering.
        set_user_ids(user_ids): Sets the user IDs for data filtering.
        set_y_bounds(y_lower, y_upper): Sets the Y-axis bounds for visualization.
        set_combine_plots(combine_plots): Sets whether to combine multiple users' data
                                          into a single plot.
        create_static_plot(flattened_fhir_dataframe): Creates and displays a static plot
                                                     based on the filtered FHIR data.
    """

    def __init__(self):
        super().__init__()
        self.start_date = None
        self.end_date = None
        self.user_ids = None
        self.y_lower = 50
        self.y_upper = 1000
        self.combine_plots = True

    def set_date_range(self, start_date: str, end_date: str):
        """Sets the start and end dates for filtering the FHIR data before visualization."""
        self.start_date = start_date
        self.end_date = end_date

    def set_user_ids(self, user_ids: list[str]):
        """Sets the list of user IDs to filter the FHIR data for visualization."""
        self.user_ids = user_ids

    def set_y_bounds(self, y_lower: float, y_upper: float):
        """Sets the lower and upper bounds for the Y-axis of the plot."""
        self.y_lower = y_lower
        self.y_upper = y_upper

    def set_combine_plots(self, combine_plots: bool):
        """Determines whether to combine plots from multiple users into a single plot or create
        separate plots for each.
        """
        self.combine_plots = combine_plots

    def create_static_plot(
        self, flattened_fhir_dataframe: FHIRDataFrame
    ) -> plt.Figure | None:
        """
        Generates a static plot based on the filtered FHIR data. This function leverages existing
        data selection utilities to filter the data by user and/or date range before plotting.
        Depending on the class configuration, it either generates a combined plot for all specified
        users or individual plots for each user.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame containing the data
                to be visualized.

        Returns:
            plt.Figure: The matplotlib figure object of the plot if a single plot is generated.
                Returns None if separate plots are created for each user or if no plot is generated
                due to errors.
        """
        fig = None
        # Filter by date range if both start and end dates are provided
        if self.start_date and self.end_date:
            flattened_fhir_dataframe = select_data_by_dates(
                flattened_fhir_dataframe, self.start_date, self.end_date
            )

        # Determine users to plot
        users_to_plot = (
            self.user_ids
            if self.user_ids
            else flattened_fhir_dataframe.df[ColumnNames.USER_ID.value].unique()
        )

        if self.combine_plots:
            fig = self.plot_combined(flattened_fhir_dataframe, users_to_plot)
        else:
            fig = self.plot_individual(flattened_fhir_dataframe, users_to_plot)

        return fig

    def plot_combined(
        self, flattened_fhir_dataframe: FHIRDataFrame, users_to_plot
    ) -> plt.Figure:
        """
        Generates a combined static plot for multiple users. Each user's data is aggregated and
        plotted within the same figure. This method is called when the class is configured to
        combine plots.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): The filtered FHIRDataFrame to be visualized.
            users_to_plot (Iterable): A collection of user IDs indicating whose data should be
                included in the plot.

        Returns:
            plt.Figure: The matplotlib figure object representing the combined plot.
        """
        plt.figure(figsize=(10, 6), dpi=300)
        for user_id in users_to_plot:
            user_df = select_data_by_user(flattened_fhir_dataframe, user_id).df
            aggregated_data = (
                user_df.groupby(ColumnNames.EFFECTIVE_DATE_TIME.value)[
                    ColumnNames.QUANTITY_VALUE.value
                ]
                .sum()
                .reset_index()
            )
            plt.bar(
                aggregated_data[ColumnNames.EFFECTIVE_DATE_TIME.value],
                aggregated_data[ColumnNames.QUANTITY_VALUE.value],
                edgecolor="black",
                linewidth=1.5,
                label=f"User {user_id}",
            )

        plt.ylim(self.y_lower, self.y_upper)
        plt.legend()
        plt.title(
            f"{flattened_fhir_dataframe.df[ColumnNames.QUANTITY_NAME.value].iloc[0]}"
            f"from {self.start_date} to {self.end_date}"
        )
        plt.xlabel("Date")
        plt.ylabel(
            f"{flattened_fhir_dataframe.df[ColumnNames.QUANTITY_NAME.value].iloc[0]}"
            " "
            f"({flattened_fhir_dataframe.df[ColumnNames.QUANTITY_UNIT.value].iloc[0]})"
        )
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig = plt.gcf()
        plt.show()
        return fig

    def plot_individual(
        self, flattened_fhir_dataframe: FHIRDataFrame, users_to_plot
    ) -> plt.Figure:
        """
        Generates individual static plots for each specified user. For each user, their data is
        aggregated and plotted in a separate figure. This method is called when the class is
        configured to generate separate plots for each user.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): The filtered FHIRDataFrame to be visualized.
            users_to_plot (Iterable): A collection of user IDs indicating whose data should be
                visualized in individual plots.

        Returns:
            plt.Figure: The matplotlib figure object of the last plot if individual plots are
                generated for multiple users. Returns None if no plot is generated due to errors.
        """
        fig = None
        for user_id in users_to_plot:
            user_fhir_df = select_data_by_user(flattened_fhir_dataframe, user_id)
            plt.figure(figsize=(10, 6), dpi=300)
            aggregated_data = (
                user_fhir_df.df.groupby(ColumnNames.EFFECTIVE_DATE_TIME.value)[
                    ColumnNames.QUANTITY_VALUE.value
                ]
                .sum()
                .reset_index()
            )
            plt.bar(
                aggregated_data[ColumnNames.EFFECTIVE_DATE_TIME.value],
                aggregated_data[ColumnNames.QUANTITY_VALUE.value],
                edgecolor="black",
                linewidth=1.5,
                label=f"User {user_id}",
            )
            plt.legend()
            plt.title(
                f"{user_fhir_df.df[ColumnNames.QUANTITY_NAME.value].iloc[0]} for User {user_id}"
                f"from {self.start_date} to {self.end_date}"
            )
            plt.xlabel("Date")
            plt.ylabel(
                f"{user_fhir_df.df[ColumnNames.QUANTITY_NAME.value].iloc[0]}"
                " "
                f"({user_fhir_df.df[ColumnNames.QUANTITY_UNIT.value].iloc[0]})"
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            if len(users_to_plot) == 1:
                fig = plt.gcf()
            plt.show()
        return fig
