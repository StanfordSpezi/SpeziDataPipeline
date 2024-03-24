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
from data_flattening.FHIR_data_flattener import FHIRDataFrame
from data_visualization.data_visualizer import DataVisualizer


class DataExporter(DataVisualizer):
    def __init__(self, flattened_FHIRDataFrame: FHIRDataFrame):
        super().__init__()
        self.flattened_FHIRDataFrame = flattened_FHIRDataFrame
        # Default values
        self.start_date = "2022-01-01"
        self.end_date = "2022-12-31"
        self.user_ids = None
        self.y_lower = 50
        self.y_upper = 100
        self.same_plot = False
        self.dpi = 300

    def export_to_csv(self, filename):
        self.flattened_FHIRDataFrame.df.to_csv(filename, index=False)

    def create_and_save_plot(self, filename):
        """Generates a plot using inherited create_static_plot method and saves it."""
        try:
            if self.user_ids is None or len(self.user_ids) > 1:
                print("Select a single user for enabling figure saving.")
            else:
                fig = super().create_static_plot(self.flattened_FHIRDataFrame)
                fig.savefig(filename, dpi=self.dpi)
                print("Plot saved successfully.")

        except (TypeError, ValueError) as e:
            print(f"An error occurred while generating the plot: {e}")
