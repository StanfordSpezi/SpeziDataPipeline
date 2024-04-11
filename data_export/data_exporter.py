#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module encompasses the DataExporter class, an extension of the DataVisualizer class,
tailored for the exportation and visualization of FHIR (Fast Healthcare Interoperability Resources)
data. The DataExporter class enhances the capability to work with flattened FHIR data by
facilitating the exportation of such data into CSV format and the generation of visual plots from
this data catering to varied analysis and reporting needs.

The class is designed to support a wide range of functionalities for manipulating, filtering,
and visualizing FHIR data. Users can specify parameters such as date ranges and user IDs to narrow
down the data of interest. Additionally, the class offers customization options for the visual
representation of data, including deciding whether to combine data from multiple users.

Key Features:
- Exporting Filtered FHIR Data: Allows for the exportation of FHIR data filtered by custom
  parameters (e.g., date range, user IDs) to CSV files, facilitating further data analysis or
  archiving.
- Enhanced Visualization Capabilities: Supports the generation of plots from FHIR data, with
  options to customize plot aesthetics such as Y-axis bounds. The class can handle both general
  observation data and ECG-specific data, offering specialized plotting functions for ECG
  waveforms.
- Parameter Customization: Users can set various parameters (e.g., start and end dates, user IDs,
  Y-axis bounds) to tailor the data exportation and visualization processes to specific
  requirements, enhancing the utility and flexibility of data analysis workflows.
- Support for ECG Data: Includes specialized functionalities for visualizing ECG (electrocardiogram)
  data, making it a valuable tool for healthcare data analysts and researchers focusing on
  cardiac health.

The DataExporter class builds on the foundational capabilities of the DataVisualizer class,
providing a seamless interface for users to both visualize and export FHIR data with ease. It is
a crucial component of the data analysis pipeline, offering streamlined processes for handling
FHIR data.

Example Usage:
    # Assuming fhir_dataframe is an instance of FHIRDataFrame containing the data to be
    visualized/exported
    data_exporter = DataExporter(fhir_dataframe)
    data_exporter.set_date_range("2021-01-01", "2021-01-31")
    data_exporter.set_user_ids(["user1", "user2"])
    data_exporter.export_to_csv("filtered_data.csv")
    data_exporter.create_and_save_plot("data_visualization.png")
"""

# Local application/library specific imports
from data_flattening.fhir_resources_flattener import FHIRDataFrame, FHIRResourceType
from data_visualization.data_visualizer import (
    DataVisualizer,
    ECGVisualizer,
    DEFAULT_DPI_VALUE,
)


class DataExporter(DataVisualizer, ECGVisualizer):  # pylint: disable=unused-variable
    """
    Extends DataVisualizer to provide functionalities for exporting FHIR data to CSV
    files and saving visualizations as image files. This class enables data export
    and visualization with customized parameters such as date ranges, user IDs, and
    Y-axis bounds, catering to specific requirements for data analysis and reporting.

    Attributes:
        flattened_fhir_dataframe (FHIRDataFrame): A flattened FHIRDataFrame intended
            for export or visualization.
        start_date (str, optional): Start date for data filtering; used to define the
            beginning of the date range of interest.
        end_date (str, optional): End date for data filtering; used to define the end
            of the date range of interest.
        user_ids (List[str], optional): List of user IDs for filtering the data. If None,
            data for all users is considered.
        y_lower (int): Lower bound for the Y-axis of the plots.
        y_upper (int): Upper bound for the Y-axis of the plots.
        combine_plots (bool): Indicates whether to combine data from multiple users into
            a single plot. Defaults to False to create separate plots for each user.
    """

    def __init__(self, flattened_fhir_dataframe: FHIRDataFrame):
        """
        Initializes the DataExporter with the given FHIRDataFrame and default parameters
        for data export and visualization.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame to be used for
                data export and visualization.
        """
        super().__init__()
        self.flattened_fhir_dataframe = flattened_fhir_dataframe
        self.start_date = None
        self.end_date = None
        self.user_ids = None
        self.y_lower = None
        self.y_upper = None
        self.combine_plots = False

    def export_to_csv(self, filename):
        """
        Exports the filtered FHIR data to a CSV file. Filtering is based on the
        specified date range and user IDs.

        Parameters:
            filename (str): The filename or path where the CSV file will be saved.
        """
        self.flattened_fhir_dataframe.df.to_csv(filename, index=False)

    def create_filename(self, base_filename, user_id, idx=None):
        """
        Constructs a filename incorporating the user ID and date or date range. If an
        index is provided, it is appended to ensure uniqueness of filenames, especially
        useful when saving multiple plots for the same parameters.

        Parameters:
            base_filename (str): The base filename or path for saving the image.
            user_id (str): User ID to include in the filename for personalized exports.
            idx (int, optional): Index to append to the filename for distinguishing
                between multiple plots.

        Returns:
            str: The constructed filename including the user ID and date or date range.
        """
        date_range = "all_dates"
        if self.start_date and self.end_date:
            date_range = f"{self.start_date}_to_{self.end_date}"
        elif self.start_date:
            date_range = f"from_{self.start_date}"
        elif self.end_date:
            date_range = f"to_{self.end_date}"

        # Construct the filename with optional index for multiple figures
        filename_parts = [base_filename.rstrip(".png"), f"user_{user_id}", date_range]
        if idx is not None:
            filename_parts.append(f"fig{idx}")
        filename = "_".join(filename_parts) + ".png"

        return filename

    def create_and_save_plot(self, base_filename: str):
        """
        Generates and saves plots for the FHIR data based on the set parameters, including
        date range and user IDs. The method decides between static plots or ECG subplots
        depending on the data type and saves the generated plots using customized filenames.

        Parameters:
            base_filename (str): The base filename or path for saving the plot images.

        Note:
            This method dynamically handles the creation and saving of either a single plot
            or multiple plots based on the filtering parameters. It supports customization
            through class attributes set prior to calling this method.
        """
        user_ids = (
            self.user_ids if self.user_ids else [None]
        )  # Handle None as a case for all users

        # Assuming that figs can either be a single figure or
        # a list of (list of figures, user_id) tuples
        figs = []
        if self.flattened_fhir_dataframe.resource_type in [
            FHIRResourceType.OBSERVATION,
            FHIRResourceType.ECG_OBSERVATION,
        ]:
            for user_id in user_ids:
                if (
                    self.flattened_fhir_dataframe.resource_type
                    == FHIRResourceType.OBSERVATION
                ):
                    data_visualizer = DataVisualizer()
                    data_visualizer.set_user_ids(
                        [user_id]
                    )  # Filter for one user at a time if multiple are provided
                    fig_list = data_visualizer.create_static_plot(
                        self.flattened_fhir_dataframe
                    )
                    if fig_list:
                        figs.extend([(fig, user_id) for fig in fig_list])
                else:  # FHIRResourceType.ECG_OBSERVATION
                    ecg_data_visualizer = ECGVisualizer()
                    ecg_data_visualizer.set_user_ids(
                        [user_id]
                    )  # Filter for one user at a time if multiple are provided
                    fig_list = ecg_data_visualizer.plot_ecg_subplots(
                        self.flattened_fhir_dataframe
                    )
                    if fig_list:
                        figs.extend([(fig, user_id) for fig in fig_list])

        for idx, (fig, user_id) in enumerate(figs, start=1):
            filename = self.create_filename(base_filename, user_id, idx)
            fig.savefig(filename, dpi=DEFAULT_DPI_VALUE)
            print(f"Plot saved successfully to: {filename}")
        if not figs:
            print("No plots were generated.")
