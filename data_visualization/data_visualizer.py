#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
The ECG Visualization Module

This Python module provides specialized tools for the visualization of ECG (Electrocardiogram)
data, leveraging the Fast Healthcare Interoperability Resources (FHIR) format. It is designed to
facilitate the analysis and interpretation of ECG data by healthcare professionals, researchers,
and data analysts. By extending foundational visualization capabilities, this module introduces
specific functionalities tailored to the nuances of ECG data, including individual lead
visualization, date and user filtering, and customizable plotting aesthetics.

Main Features:
- ECG Data Processing: Supports the import and preprocessing of ECG data in FHIR format, preparing
  it for visualization.
- User and Date Filtering: Allows users to filter ECG observations by specific patient IDs and date
  ranges, enabling focused analysis on relevant data subsets.
- Custom Plotting: Offers specialized plotting functions for ECG data, including the visualization
  of individual ECG leads in separate subplots and the adjustment of plot aesthetics such as axis
  properties.
- Combined and Individual Plots: Facilitates the creation of both combined plots (for comparative
  analysis across multiple patients) and individual plots (for detailed examination of a single
  patient's ECG data).
- Visualization Customization: Provides options to set Y-axis bounds, combine or separate plots for
  multiple users, and customize line widths and tick intervals for ECG waveforms.

Usage Example:
from ecg_visualization_module import ECGVisualizer, FHIRDataFrame

# Load your ECG data into a FHIRDataFrame
fhir_dataframe = FHIRDataFrame(load_your_data_here())

# Initialize the ECG visualizer
ecg_visualizer = ECGVisualizer()

# Set visualization parameters
ecg_visualizer.set_user_ids(['user123'])
ecg_visualizer.set_date_range('2021-01-01', '2021-01-31')
ecg_visualizer.set_y_bounds(-1.0, 1.0)

# Generate and display the ECG plots
ecg_visualizer.plot_ecg_subplots(fhir_dataframe)
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
    select_data_by_dates,
)
from data_flattening.fhir_resources_flattener import (
    FHIRResourceType,
    FHIRDataFrame,
    ColumnNames,
)

TIME_UNIT = "sec"
ECG_UNIT = "uV"
DEFAULT_SAMPLE_RATE_VALUE = 500
DEFAULT_DPI_VALUE = 300
DEFAULT_LINE_WIDTH_VALUE = 0.5
DEFAULT_AMPLITUDE_ECG = 1.8
DEFAULT_TIME_TICKS = 0.2


class DataVisualizer:  # pylint: disable=unused-variable
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
        y_lower (float): Lower bound for the Y-axis. Defaults to 50.
        y_upper (float): Upper bound for the Y-axis. Defaults to 1000.
        combine_plots (bool): If True, combines data from multiple users into a single plot.
            Defaults to True.
    """

    def __init__(self):
        """Initializes the DataVisualizer with default parameters for data visualization."""
        super().__init__()
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
            fhir_dataframe (FHIRDataFrame): The FHIRDataFrame containing the data
                to be visualized.

        Returns:
            list: A list of matplotlib figure objects of the generated plots.
        """
        figures = []

        if self.user_ids is None:
            user_ids = fhir_dataframe.df[ColumnNames.USER_ID.value].unique()
        else:
            user_ids = (
                self.user_ids if isinstance(self.user_ids, list) else [self.user_ids]
            )

        if self.start_date and self.end_date:
            fhir_dataframe = select_data_by_dates(
                fhir_dataframe, self.start_date, self.end_date
            )

        if fhir_dataframe.df.empty:
            print("No data for the selected date range.")
            return figures

        loinc_codes = fhir_dataframe.df[ColumnNames.LOINC_CODE.value].unique()

        for loinc_code in loinc_codes:
            df_loinc = fhir_dataframe.df[
                fhir_dataframe.df[ColumnNames.LOINC_CODE.value] == loinc_code
            ]

            if self.combine_plots:
                if fig := self.plot_combined(df_loinc, user_ids, loinc_code):
                    figures.append(fig)
            else:
                for user_id in user_ids:
                    if fig := self.plot_individual(df_loinc, user_id, loinc_code):
                        figures.append(fig)

        return figures

    def plot_combined(self, df_loinc, users_to_plot, loinc_code) -> plt.Figure:
        """
        Generates a combined static plot for multiple users. Each user's data
        is aggregated and plotted within the same figure. This method is useful when
        the class is configured to combine plots from multiple users into a single
        visualization for comparative analysis.

        Parameters:
            df_loinc (DataFrame): A DataFrame filtered for a specific LOINC code.
            users_to_plot (list[str]): A list of user IDs to include in the plot.
            loinc_code (str): The LOINC code that the plot is focusing on.

        Returns:
            matplotlib.figure.Figure: The figure object representing the combined plot.
        """
        plt.figure(figsize=(10, 6), dpi=DEFAULT_DPI_VALUE)

        for user_id in users_to_plot:
            user_df = df_loinc[df_loinc[ColumnNames.USER_ID.value] == user_id]
            _ = plot_data_based_on_condition(user_df, user_id)

        date_range_title = (
            "for all dates"
            if not self.start_date and not self.end_date
            else f"from {self.start_date} to {self.end_date}"
        )
        plt.title(
            f"{df_loinc[ColumnNames.QUANTITY_NAME.value].iloc[0]} "
            f"for LOINC Code {loinc_code} {date_range_title}"
        )
        plt.xlabel("Date")
        plt.ylabel(
            f"{df_loinc[ColumnNames.QUANTITY_NAME.value].iloc[0]} "
            f"({df_loinc[ColumnNames.QUANTITY_UNIT.value].iloc[0]})"
        )
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylim(self.y_lower, self.y_upper)
        plt.tight_layout()

        fig = plt.gcf()
        plt.show()
        return fig

    def plot_individual(self, df_loinc, user_id, loinc_code) -> plt.Figure:
        """
        Generates individual static plots for each specified user. For each user, their
        data is aggregated and plotted in a separate figure. This method is called to
        generate detailed plots for individual users, focusing on data related to a
        specific LOINC code.

        Parameters:
            df_loinc (DataFrame): A DataFrame filtered for a specific LOINC code.
            user_id (str): The user ID for which to generate the plot.
            loinc_code (str): The LOINC code that the plot is focusing on.

        Returns:
            matplotlib.figure.Figure: The figure object representing the individual plot.
            If no data is found for the specified user ID and LOINC code, returns None.
        """
        if user_id is None:
            print("User ID must be provided for individual plots.")
            return None

        user_df = df_loinc[df_loinc[ColumnNames.USER_ID.value] == user_id]
        if user_df.empty:
            print(f"No data found for user ID {user_id} and LOINC code {loinc_code}.")
            return None

        plt.figure(figsize=(10, 6), dpi=DEFAULT_DPI_VALUE)
        date_range_title = (
            "for all dates"
            if not self.start_date and not self.end_date
            else f"from {self.start_date} to {self.end_date}"
        )
        plt.title(
            f"{user_df[ColumnNames.QUANTITY_NAME.value].iloc[0]} "
            f"for User ID {user_id} {date_range_title}"
        )

        _ = plot_data_based_on_condition(user_df, user_id)

        plt.xlabel("Date")
        plt.ylabel(
            f"{user_df[ColumnNames.QUANTITY_NAME.value].iloc[0]} "
            f"({user_df[ColumnNames.QUANTITY_UNIT.value].iloc[0]})"
        )
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylim(self.y_lower, self.y_upper)
        plt.tight_layout()

        fig = plt.gcf()
        plt.show()
        return fig


def plot_data_based_on_condition(user_df, user_id):
    """
    Dynamically plots data using either plt.scatter or plt.bar based on the condition
    of duplicate EffectiveDateTime entries for a user. Utilizes scatter plots for datasets
    with duplicate timestamps and bar plots for datasets with unique timestamps, allowing
    for appropriate visualization of the data distribution.

    Parameters:
        user_df (pd.DataFrame): The DataFrame containing data for a specific user.
        user_id (str): The ID of the user for which the data is being plotted.

    Returns:
        dict: Information about the plot, including the chosen plot type ('scatter' or 'bar')
              and the DataFrame used for plotting.
    """
    if user_df.duplicated(subset=[ColumnNames.EFFECTIVE_DATE_TIME.value]).any():
        plot_type = "scatter"
        plot_function = plt.scatter
    else:
        plot_type = "bar"
        plot_function = plt.bar

    plot_function(
        user_df[ColumnNames.EFFECTIVE_DATE_TIME.value],
        user_df[ColumnNames.QUANTITY_VALUE.value],
        label=f"User {user_id}",
        edgecolor="black",
        linewidth=1.5,
    )

    return {"plot_type": plot_type, "data_frame": user_df}


class ECGVisualizer:  # pylint: disable=unused-variable
    """
    A visualization tool for electrocardiogram (ECG) data that extends the DataVisualizer class.
    This class provides specialized plotting functions to render ECG waveforms from FHIR data frames
    that contain ECG observations for individual patients.

    The class supports plotting individual ECG leads in separate subplots, configuring axis
    properties for ECG-specific visualization needs, and filtering ECG data by user ID and effective
    datetime.
    """

    def __init__(self):
        """
        Initializes the ECGVisualizer with default parameters for ECG data visualization.
        Sets line width, date range, user IDs, amplitude scale, and time ticks for plotting.
        """
        self.lwidth = DEFAULT_LINE_WIDTH_VALUE
        self.start_date = None
        self.end_date = None
        self.user_ids = None
        self.amplitude_ecg = DEFAULT_AMPLITUDE_ECG
        self.time_ticks = DEFAULT_TIME_TICKS

    def set_date_range(self, start_date: str, end_date: str):
        """
        Sets the start and end dates for filtering the FHIR data before visualization,
        aiming to narrow down the plotted data to a specific timeframe.

        Parameters:
            start_date (str): The start date of the range for data filtering.
            end_date (str): The end date of the range for data filtering.
        """
        self.start_date = (
            datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
        )
        self.end_date = (
            datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None
        )

    def set_user_ids(self, user_ids: list[str]):
        """
        Sets the list of user IDs to filter the FHIR data for visualization,
        allowing for targeted analysis of ECG data for specific individuals.

        Parameters:
            user_ids (list[str]): A list of user IDs for data filtering.
        """
        self.user_ids = user_ids if isinstance(user_ids, list) else [user_ids]

    def _ax_plot(self, ax, x, y, secs):
        """
        Configures the axes for plotting ECG data on a given matplotlib axis object,
        setting up visual aspects like ticks, grid lines, and axis limits.

        Parameters:
            ax (matplotlib.axes.Axes): The axes object to configure for ECG plotting.
            x (np.ndarray): The array of time values for the ECG data points.
            y (np.ndarray): The array of amplitude values for the ECG data points.
            secs (int): The total duration of the ECG recording in seconds.
        """
        ax.set_xticks(np.arange(0, secs + self.time_ticks, self.time_ticks))
        ax.set_yticks(
            np.arange(-ceil(self.amplitude_ecg), ceil(self.amplitude_ecg), 1.0)
        )
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(-self.amplitude_ecg, self.amplitude_ecg)
        ax.set_xlim(0, secs)
        ax.grid(
            which="major",
            linestyle="-",
            linewidth=f"{DEFAULT_LINE_WIDTH_VALUE}",
            color="red",
        )
        ax.grid(
            which="minor",
            linestyle="-",
            linewidth=f"{DEFAULT_LINE_WIDTH_VALUE}",
            color=(1, 0.7, 0.7),
        )
        ax.plot(x, y, linewidth=self.lwidth)

    def plot_single_user_ecg(self, user_data, user_id):
        """
        Plots ECG data for a single user, generating separate subplots for each ECG lead.
        The method aims to visualize individual ECG waveforms with specified aesthetics.

        Parameters:
            user_data (DataFrame): The subset of FHIR data frame containing ECG observations
                for a specific user. The unit ECG observations must be in mV.
            user_id (str): The ID of the user whose ECG data is being plotted.

        Returns:
            list: A list of matplotlib.figure.Figure objects representing the generated ECG plots.
        """
        figures = []
        for _, row in user_data.iterrows():
            fig, axs = plt.subplots(3, 1, figsize=(15, 6), constrained_layout=True)
            effective_date = row[ColumnNames.EFFECTIVE_DATE_TIME.value].strftime(
                "%Y-%m-%d"
            )

            for i, key in enumerate(
                [
                    ColumnNames.ECG_RECORDING1.value,
                    ColumnNames.ECG_RECORDING2.value,
                    ColumnNames.ECG_RECORDING3.value,
                ]
            ):
                if row[key] is not None:
                    ecg_string = row[key]
                    ecg = (
                        np.array(ecg_string.split(), dtype=float) / 1000
                    )  # Convert unit to uV
                    sample_rate = row.get(
                        ColumnNames.SAMPLING_FREQUENCY.value, DEFAULT_SAMPLE_RATE_VALUE
                    )
                    title = f"{key} for {user_id} on {effective_date}"
                    self._plot_single_lead_ecg(ecg, sample_rate, title, axs[i])
                else:
                    axs[i].text(
                        0.5,
                        0.5,
                        "No ECG data available for this recording.",
                        ha="center",
                        va="center",
                        transform=axs[i].transAxes,
                    )
            plt.show()
            figures.append(fig)
        return figures

    def plot_ecg_subplots(self, fhir_dataframe):
        """
        Generates ECG subplots for specified users within a given date range, based on
        the filtering parameters set. This method orchestrates the creation of ECG
        visualizations for one or multiple users.

        Parameters:
            fhir_dataframe (FHIRDataFrame): The FHIRDataFrame containing ECG observation
                data to be visualized. The unit ECG observations must be in mV.

        Returns:
            list: A list of matplotlib.figure.Figure objects representing the ECG plots
                for the specified users.
        """
        figures = []

        # Determine the users to plot: if self.user_ids is None, plot for all users
        users_to_plot = (
            self.user_ids
            if self.user_ids is not None
            else fhir_dataframe.df[ColumnNames.USER_ID.value].unique()
        )
        for user_id in users_to_plot:
            # Filter data based on user_id and date range
            if self.start_date and self.end_date:
                user_data = fhir_dataframe.df[
                    (fhir_dataframe.df[ColumnNames.USER_ID.value] == user_id)
                    & (
                        fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
                        >= self.start_date
                    )
                    & (
                        fhir_dataframe.df[ColumnNames.EFFECTIVE_DATE_TIME.value]
                        <= self.end_date
                    )
                ]
            else:
                user_data = fhir_dataframe.df[
                    fhir_dataframe.df[ColumnNames.USER_ID.value] == user_id
                ]

            if not user_data.empty:
                figures.extend(self.plot_single_user_ecg(user_data, user_id))
            else:
                print(
                    f"No ECG data found for user ID: {user_id} in the given date range"
                )

        return figures

    def _plot_single_lead_ecg(
        self, ecg, sample_rate=DEFAULT_SAMPLE_RATE_VALUE, title="ECG", ax=None
    ):
        """
        Helper function to plot a single lead ECG waveform on a specified axes object.
        Configures the plot with given ECG data, sample rate, and title, enhancing the
        visual representation.

        Parameters:
            ecg (np.ndarray): The ECG waveform data points to plot.
            sample_rate (int, optional): The sample rate of the ECG recording, defaulting
                to a standard value.
            title (str, optional): The title for the subplot, defaulting to "ECG".
            ax (matplotlib.axes.Axes, optional): The axes object on which to plot the ECG.
                If None, a new axes will be created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 2))

        ax.set_title(title)
        ax.set_ylabel(f"ECG ({ECG_UNIT})")
        ax.set_xlabel(f"Time ({TIME_UNIT})")

        if isinstance(sample_rate, Decimal):
            sample_rate = float(sample_rate)

        seconds = len(ecg) / sample_rate
        step = 1.0 / sample_rate
        self._ax_plot(ax, np.arange(0, len(ecg) * step, step), ecg, seconds)


def visualizer_factory(fhir_dataframe):  # pylint: disable=unused-variable
    """
    Factory function to create a visualizer based on the resource_type attribute of
    FHIRDataFrame.

    Parameters:
        fhir_dataframe (FHIRDataFrame): An instance of FHIRDataFrame containing the
            data and resource_type attribute.

    Returns:
        An instance of either DataVisualizer or ECGVisualizer based on the
            resource_type.
    """
    if fhir_dataframe.resource_type == FHIRResourceType.OBSERVATION:
        return DataVisualizer()
    if fhir_dataframe.resource_type == FHIRResourceType.ECG_OBSERVATION:
        return ECGVisualizer()
    raise ValueError(f"Unsupported resource type: {fhir_dataframe.resource_type}")
