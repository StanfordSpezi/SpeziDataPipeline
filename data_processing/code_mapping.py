#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module contains the CodeProcessor, a utility class that manages mappings between LOINC codes
and their corresponding processing functions and value ranges. It facilitates the dynamic
association of health metrics with their processing logic, enhancing the module's flexibility
and extensibility.
"""

# Related third-party imports
from dataclasses import dataclass

# Local application/library specific imports
from .observation_processing import calculate_daily_data


@dataclass
class CodeProcessor:  # pylint: disable=unused-variable
    """
    Manages mappings between LOINC codes (and similar healthcare identifiers) and their respective
    processing details. This class serves as a central repository for the mappings of LOINC codes
    to their display names, processing functions, and default value ranges for outlier filtering.

    The `CodeProcessor` is essential for the `FHIRDataProcessor` to dynamically associate specific
    health metrics with their processing logic and acceptable value ranges, thereby enabling
    tailored data analysis, normalization, and outlier detection based on the type of health data
    being processed.

    Attributes:
        code_mappings (dict): A dictionary mapping health data identifiers (e.g., LOINC codes) to
                              tuples containing their display name, code, and system URI. This
                              mapping supports the interpretability and traceability of the
                              processed health metrics.
        code_to_function (dict): Maps health data identifiers to their respective data processing
                                 functions, allowing for flexible and dynamic application of
                                 specific processing routines based on the health metric being
                                 analyzed.
        default_value_ranges (dict): Specifies the default ranges of acceptable values for
                                     different health metrics, aiding in the filtering of
                                     outliers from the dataset. These ranges are keyed by
                                     health data identifiers and are used during the
                                     preprocessing phase to ensure data quality.
    """

    def __init__(self):
        """
        Initializes the CodeProcessor with predefined mappings for LOINC codes to their display
        names, codes, and systems, as well as associations between these codes and specific
        processing functions and default value ranges for outlier filtering.
        """
        # Maps LOINC codes and similar identifiers to their display names, codes, and systems
        self.code_mappings = {
            "9052-2": ("Calorie intake total", "9052-2", "http://loinc.org"),
            "55423-8": (
                "Number of steps in unspecified time Pedometer",
                "55423-8",
                "http://loinc.org",
            ),
            "HKQuantityTypeIdentifierDietaryProtein": (
                "Dietary Protein",
                "HKQuantityTypeIdentifierDietaryProtein",
                "http://developer.apple.com/documentation/healthkit",
            ),
            "131328": (
                "MDC_ECG_ELEC_POTL",
                "131328",
                "urn:oid:2.16.840.1.113883.6.24",
            ),
        }

        # Maps LOINC codes and similar identifiers to processing functions
        self.code_to_function = {
            "9052-2": calculate_daily_data,
            "55423-8": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryProtein": calculate_daily_data,
        }

        # Example of ranges that could be used for filtering
        self.default_value_ranges = {
            "9052-2": (0, 2700),
            "55423-8": (0, 30000),
            # Add other code ranges as needed
        }
