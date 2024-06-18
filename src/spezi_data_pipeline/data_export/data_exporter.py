#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module extends functionality from the Spezi Data Pipeline for exporting and visualizing FHIR
data, specifically focusing on DataExplorer and ECGExplorer functionalities and exporting data
to CSV files and image files.

Users can specify parameters such as date ranges and user IDs to narrow down the data of interest.
Additionally, the class offers customization options for the visual representation of data,
including deciding whether to combine data from multiple users.

Classes:
- `DataExporter`: Extends `DataExplorer` and `ECGExplorer` to facilitate exporting filtered FHIR
                  data to CSV files and generating and saving plots as image files. It allows
                  customization of date ranges, user IDs, and plot parameters for data analysis
                  and reporting.

Functions:
- No standalone functions in this module.
"""

# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    FHIRResourceType,
)
from spezi_data_pipeline.data_exploration.data_explorer import (
    DataExplorer,
    ECGExplorer,
    DEFAULT_DPI_VALUE,
)


class DataExporter(DataExplorer, ECGExplorer):  # pylint: disable=unused-variable
    """
    Extends DataExporter to provide functionalities for exporting FHIR data to CSV
    files and saving explorations as image files. This class enables data export
    and exploration with customized parameters such as date ranges, user IDs, and
    Y-axis bounds, catering to specific requirements for data analysis and reporting.

    Attributes:
        flattened_fhir_dataframe (FHIRDataFrame): A flattened FHIRDataFrame intended
            for export or exploration.
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
        for data export and exploration.

        Parameters:
            flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame to be used for
                data export and exploration.
        """
        super().__init__()
        self.flattened_fhir_dataframe = flattened_fhir_dataframe

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
        if not self.user_ids:
            return  # Do not proceed with plot creation.

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
                    data_visualizer = DataExplorer()
                    data_visualizer.set_user_ids(
                        [user_id]
                    )  # Filter for one user at a time if multiple are provided
                    fig_list = data_visualizer.create_static_plot(
                        self.flattened_fhir_dataframe
                    )
                    if fig_list:
                        figs.extend([(fig, user_id) for fig in fig_list])
                else:  # FHIRResourceType.ECG_OBSERVATION
                    ecg_data_visualizer = ECGExplorer()
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
