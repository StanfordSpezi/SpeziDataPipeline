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

# Standard library imports
from datetime import datetime
from decimal import Decimal
from math import ceil

# Related third-party imports
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# Local application/library specific imports
from data_processing.data_processor import (
    FHIRDataProcessor,
    select_data_by_dates,
    select_data_by_user,
)
from data_flattening.fhir_resources_flattener import (
    FHIRResourceType,
    FHIRDataFrame,
    ColumnNames,
)

TIME_UNIT = "sec"
ECG_UNIT = "uV"


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

        if not flattened_fhir_dataframe.resource_type == FHIRResourceType.OBSERVATION:
            print(
                f"The input FHIRDataFrame is not an {FHIRResourceType.OBSERVATION.value} type."
            )
            print(
                "See documentation for related methods for"
                f"{flattened_fhir_dataframe.resource_type.name} types."
            )
            return None

        if not isinstance(self.user_ids, list):
            print("The selected user(s) should be inputted in a list structure.")
            return None

        if not flattened_fhir_dataframe.df[ColumnNames.LOINC_CODE.value].nunique() == 1:
            print("The FHIRDataFrame should contain data of a single LOINC code.")
            return None

        if self.start_date and self.end_date:
            flattened_fhir_dataframe = select_data_by_dates(
                flattened_fhir_dataframe, self.start_date, self.end_date
            )

        if flattened_fhir_dataframe.df.empty:
            print("No data for the selected date range.")
            return None

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

            if aggregated_data.empty:
                print(f"No data found for user ID(s): {users_to_plot}")
                return None

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

            if aggregated_data.empty:
                print(f"No data found for user ID(s): {users_to_plot}")
                return None

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
                return fig
            plt.show()
        return fig


class ECGVisualizer(DataVisualizer):  # pylint: disable=unused-variable
    """
    A visualization tool for electrocardiogram (ECG) data that extends the DataVisualizer class.
    This class provides specialized plotting functions to render ECG waveforms from FHIR data frames
    that contain ECG observations for individual patients.

    The class supports plotting individual ECG leads in separate subplots, configuring axis
    properties for ECG-specific visualization needs, and filtering ECG data by user ID and effective
    datetime.
    """

    def __init__(self):
        super().__init__()
        self.lwidth = 0.5
        self.amplitude_ecg = 1.8
        self.time_ticks = 0.2

    def _ax_plot(self, ax, x, y, secs):
        """
        Configures the axes for plotting ECG data on a given matplotlib axis.

        Parameters:
            ax (matplotlib.axes.Axes): Axis to configure.
            x (np.ndarray): Time values for ECG data.
            y (np.ndarray): Amplitude values for ECG data.
            secs (int): The total duration of the ECG recording in seconds.

        This method sets major and minor ticks for the x-axis based on the duration of the ECG
        recording and the interval between major ticks. It adjusts the y-axis limits based on the
        expected maximum amplitude of the ECG waveform. Additionally, it enables grid lines and
        plots the ECG waveform.
        """
        ax.set_xticks(np.arange(0, secs + self.time_ticks, self.time_ticks))
        ax.set_yticks(
            np.arange(-ceil(self.amplitude_ecg), ceil(self.amplitude_ecg), 1.0)
        )
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(-self.amplitude_ecg, self.amplitude_ecg)
        ax.set_xlim(0, secs)
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")
        ax.grid(which="minor", linestyle="-", linewidth="0.5", color=(1, 0.7, 0.7))
        ax.plot(x, y, linewidth=self.lwidth)

    def plot_ecg_subplots(self, fhir_dataframe, user_id: str, effective_datetime: str):
        """
        Plots three ECG recordings for a specific user from a FHIRDataFrame, filtering the data
        by the effective date-time.

        Parameters:
            fhir_dataframe (FHIRDataFrame): The FHIRDataFrame containing the ECG observation data.
            user_id (str): The ID of the user for whom to plot the ECG data.
            effective_datetime (str): The effective date-time string used to filter the ECG data.

        Returns:
            matplotlib.figure.Figure: The figure object containing the ECG plots.
            Returns None if no data is found.
        """
        self.set_date_range(effective_datetime, effective_datetime)
        self.set_user_ids([user_id])

        effective_datetime = datetime.strptime(effective_datetime, "%Y-%m-%d").date()
        user_data = fhir_dataframe.df[
            (fhir_dataframe.df[ColumnNames.USER_ID.value] == user_id)
            & (
                fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
                == effective_datetime
            )
        ]

        if user_data.empty:
            print(f"No ECG data found for user ID: {user_id} at {effective_datetime}")
            return None

        fig, axs = plt.subplots(3, 1, figsize=(15, 6), constrained_layout=True)

        for i, key in enumerate(
            [
                ColumnNames.ECG_RECORDING1.value,
                ColumnNames.ECG_RECORDING2.value,
                ColumnNames.ECG_RECORDING3.value,
            ]
        ):
            if key in user_data:
                ecg_string = user_data[key].iloc[0]
                print(ecg_string)
                ecg = np.array(ecg_string.split(), dtype=float)
                ecg = ecg / 1000  # convert unit to uV
                sample_rate = user_data.get(
                    ColumnNames.SAMPLING_FREQUENCY.value, 500
                ).iloc[0]
                title = f"{key} for {ColumnNames.USER_ID.value} {user_id} at {effective_datetime}"
                self._plot_single_lead_ecg(
                    ecg, sample_rate=sample_rate, title=title, ax=axs[i]
                )
            else:
                axs[i].text(
                    0.5,
                    0.5,
                    "No data available",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axs[i].transAxes,
                )
        plt.show()

        return fig

    def _plot_single_lead_ecg(self, ecg, sample_rate=500, title="ECG", ax=None):
        """
        Helper function to plot a single lead ECG on a specified axes object.

        Parameters:
            ecg (np.ndarray): The ECG waveform data points to plot.
            sample_rate (int, optional): The sample rate of the ECG recording. Defaults to 500.
            title (str, optional): The title for the subplot. Defaults to "ECG".
            ax (matplotlib.axes.Axes, optional): The axes object on which to plot the ECG.
                If None, a new figure and axes will be created. Defaults to None.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 2))

        ax.set_title(title)
        ax.set_ylabel(f"ECG ({ECG_UNIT})")
        # ax.set_ylabel(user_data.get("ECG", "uV").iloc[0])
        ax.set_xlabel(f"Time ({TIME_UNIT})")

        if isinstance(sample_rate, Decimal):
            sample_rate = float(sample_rate)

        seconds = len(ecg) / sample_rate
        step = 1.0 / sample_rate
        self._ax_plot(ax, np.arange(0, len(ecg) * step, step), ecg, seconds)


def visualizer_factory(fhir_dataframe):  # pylint: disable=unused-variable
    """
    Factory function to create a visualizer based on the resource_type attribute of FHIRDataFrame.

    Parameters:
    - fhir_dataframe: An instance of FHIRDataFrame containing the data and resource_type attribute.

    Returns:
    - An instance of DataVisualizer or ECGVisualizer based on the resource_type.
    """
    if fhir_dataframe.resource_type == FHIRResourceType.OBSERVATION:
        return DataVisualizer()
    if fhir_dataframe.resource_type == FHIRResourceType.ECG_OBSERVATION:
        return ECGVisualizer()
    raise ValueError(f"Unsupported resource type: {fhir_dataframe.resource_type}")
