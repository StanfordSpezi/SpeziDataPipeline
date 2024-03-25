#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

# Local application/library specific imports
from data_flattening.fhir_data_flattener import FHIRDataFrame
from data_visualization.data_visualizer import DataVisualizer


class DataExporter(DataVisualizer):
    """
    Extends the DataVisualizer class to enable data exporting functionalities, allowing
    for the export of FHIR data into CSV files and the creation and saving of visualizations
    as image files. This class handles the preparation and configuration of data for export
    based on specified parameters.

    Attributes:
        flattened_fhir_dataframe (FHIRDataFrame): The FHIRDataFrame containing flattened FHIR
                                            data for export or visualization.
        start_date (str): The start date for filtering the data, defaulting to "2022-01-01".
        end_date (str): The end date for filtering the data, defaulting to "2022-12-31".
        user_ids (List[str], optional): A list of user IDs for filtering the data.
                                    If None, all users are considered.
        y_lower (int): The lower bound for the y-axis of the plot, defaulting to 50.
        y_upper (int): The upper bound for the y-axis of the plot, defaulting to 100.
        same_plot (bool): Indicates whether to combine multiple users' data into a single plot.
                        Defaults to False.
        dpi (int): The resolution of the saved plot image, defaulting to 300 DPI.

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
        # Default values
        self.start_date = "2022-01-01"
        self.end_date = "2022-12-31"
        self.user_ids = None
        self.y_lower = 50
        self.y_upper = 100
        self.same_plot = False
        self.dpi = 300

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
            if self.user_ids is None or len(self.user_ids) > 1:
                print("Select a single user for enabling figure saving.")
            else:
                fig = super().create_static_plot(self.flattened_fhir_dataframe)
                fig.savefig(filename, dpi=self.dpi)
                print("Plot saved successfully.")

        except (TypeError, ValueError) as e:
            print(f"An error occurred while generating the plot: {e}")
