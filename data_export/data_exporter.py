#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
 This module defines the DataExporter class, which extends the DataVisualizer
 class to provide additional functionality for exporting FHIR (Fast Healthcare 
 Interoperability Resources) data. The DataExporter allows for exporting
 flattened FHIR data frames to CSV files and generating plots from the data.
 It supports setting various parameters such as date ranges, user IDs, and
 plot properties to customize  the export and visualization processes according 
 to specific requirements.
 Features include:
 - Exporting data to CSV with `export_to_csv`.
 - Creating and saving plots with customized parameters through
 `create_and_save_plot`.
 """

# Local application/library specific imports
from data_flattening.fhir_resources_flattener import FHIRDataFrame, FHIRResourceType
from data_visualization.data_visualizer import DataVisualizer, visualizer_factory


class DataExporter(DataVisualizer):  # pylint: disable=unused-variable
    """
    Extends the DataVisualizer class to enable data exporting functionalities, allowing
    for the export of FHIR data into CSV files and the creation and saving of visualizations
    as image files. This class handles the preparation and configuration of data for export
    based on specified parameters.

    Attributes:
        flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame containing flattened FHIR
                                            data for export or visualization.
        start_date (str): The start date for filtering the data.
        end_date (str): The end date for filtering the data.
        user_ids (List[str], optional): A list of user IDs for filtering the data.
                                    If None, all users are considered.
        y_lower (int): The lower bound for the y-axis of the plot.
        y_upper (int): The upper bound for the y-axis of the plot.
        combine_plots (bool): Indicates whether to combine multiple users' data into a single plot.
                        Defaults to False.
    """

    def __init__(self, flattened_fhir_dataframe: FHIRDataFrame):
        """
        Initializes the DataExporter with a specified FHIRDataFrame and default parameters
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
        Exports the FHIR data to a CSV file.

        Parameters:
            filename (str): The path or filename where the CSV file will be saved.
        """
        self.flattened_fhir_dataframe.df.to_csv(filename, index=False)

    def create_and_save_plot(self, filename):
        """
        Generates a plot for the specified FHIR data and saves it to a file. Only supports
        creating and saving plots for data related to a single user. If multiple user IDs are
        present or no user ID is specified, it prompts to select a single user.

        Parameters:
            filename (str): The path or filename where the plot image will be saved.

        Raises:
            TypeError: If an unsupported type is encountered in the plotting data.
            ValueError: If the data values are outside the expected ranges or if plotting
            parameters are incorrect.
        """
        try:
            visualizer = visualizer_factory(self.flattened_fhir_dataframe)
            if (
                self.flattened_fhir_dataframe.resource_type
                == FHIRResourceType.OBSERVATION
            ):
                if self.user_ids is None or len(self.user_ids) > 1:
                    print("Select a single user for enabling figure saving.")
                    return None

                if fig := super().create_static_plot(self.flattened_fhir_dataframe):
                    fig.savefig(filename, dpi=300)
                    print("Plot saved successfully to:", filename)
                    # return fig

            elif (
                self.flattened_fhir_dataframe.resource_type
                == FHIRResourceType.ECG_OBSERVATION
            ):
                if self.user_ids is None:
                    print("Select a single user for enabling figure saving.")
                    return None

                if not isinstance(self.user_ids, str):
                    print("The selected user to plot should be inputted as a string.")
                    return None

                visualizer.set_user_ids(self.user_ids)
                fig = visualizer.plot_ecg_subplots(
                    self.flattened_fhir_dataframe,
                    self.user_ids,
                    effective_datetime=self.start_date,
                )
                if fig:
                    fig.savefig(filename, dpi=300)
                    print("Plot saved successfully to:", filename)
                    # return fig

        except (TypeError, ValueError) as e:
            print(f"An error occurred while generating the plot: {e}")
            return None

        return None
