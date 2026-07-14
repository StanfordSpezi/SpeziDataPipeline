#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module introduces the `CodeProcessor` class, designed to handle the mapping and processing of
healthcare data identifiers, primarily LOINC codes, within the context of FHIR (Fast Healthcare
Interoperability Resources) data processing. The class plays a pivotal role in associating health
data metrics with appropriate processing functions, display names, and acceptable value ranges,
facilitating dynamic data analysis, normalization, and quality control.

The `CodeProcessor` serves as a foundational component within a larger data processing pipeline,
enabling customizable handling of diverse health metrics. By providing mappings from LOINC codes
and similar healthcare identifiers to processing details, it supports a wide range of operations,
including outlier detection, data aggregation, and trend analysis. The class enhances the
flexibility and adaptability of health data analysis workflows, making it easier to address the
specific requirements of different health metrics and data sources.

Key Features:
- Dynamic Association: Links LOINC codes and similar identifiers to their processing logic,
  allowing for tailored data analysis based on the health metric being examined.
- Customizable Processing: Facilitates the association of health metrics with specific data
  processing functions, supporting operations such as daily totals calculation, averaging,
  and moving average analysis.
- Outlier Filtering: Employs default value ranges for various health metrics to identify and
  filter outliers, ensuring data quality and reliability.

"""

# Related third-party imports
from dataclasses import dataclass

# Local application/library specific imports
from spezi_data_pipeline.data_processing.observation_processor import (
    calculate_daily_data,
    calculate_average_data,
)


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
            "8867-4": (
                "Heart rate",
                "8867-4",
                "http://loinc.org",
            ),
            "HKQuantityTypeIdentifierPhysicalEffort": (
                "Apple Physical Effort",
                "HKQuantityTypeIdentifierPhysicalEffort",
                "http://developer.apple.com/documentation/healthkit",
            ),
            "41981-2": (
                "Calories burned",
                "41981-2",
                "http://loinc.org",
            ),
            "HKQuantityTypeIdentifierVO2Max": (
                "VO2 Max",
                "HKQuantityTypeIdentifierVO2Max",
                "http://developer.apple.com/documentation/healthkit",
            ),
        }

        # Maps LOINC codes and similar identifiers to processing functions
        self.code_to_function = {
            # Summed up per day
            "9052-2": calculate_daily_data,
            "55423-8": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryProtein": calculate_daily_data,
            "41981-2": calculate_daily_data,
            "HKQuantityTypeIdentifierAppleExerciseTime": calculate_daily_data,
            "HKQuantityTypeIdentifierAppleMoveTime": calculate_daily_data,
            "HKQuantityTypeIdentifierAppleStandTime": calculate_daily_data,
            "HKQuantityTypeIdentifierBasalEnergyBurned": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryBiotin": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryCaffeine": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryCalcium": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryCarbohydrates": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryChloride": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryChoresterol": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryChromium": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryCopper": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryFatMonounsaturated": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryFatPolyunsaturated": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryFatSaturated": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryFat": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryFiber": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryFolate": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryIodine": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryIron": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryMagnesium": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryManganese": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryMolybdenum": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryNiacin": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryPantothenicAcid": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryPhosphorus": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryPotassium": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryRiboflavin": calculate_daily_data,
            "HKQuantityTypeIdentifierDietarySelenium": calculate_daily_data,
            "HKQuantityTypeIdentifierDietarySodium": calculate_daily_data,
            "HKQuantityTypeIdentifierDietarySugar": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryThiamin": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminA": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminB12": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminB6": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminC": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminD": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminE": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryVitaminK": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryWater": calculate_daily_data,
            "HKQuantityTypeIdentifierDietaryZinc": calculate_daily_data,
            "HKQuantityTypeIdentifierDistanceCycling": calculate_daily_data,
            "HKQuantityTypeIdentifierDistanceDownhillSnowSports": calculate_daily_data,
            "93816-7": calculate_daily_data,
            "HKQuantityTypeIdentifierDistanceWalkingRunning": calculate_daily_data,
            "HKQuantityTypeIdentifierDistanceWheelchair": calculate_daily_data,
            "100304-5": calculate_daily_data,
            "HKQuantityTypeIdentifierSwimmingStrokeCount": calculate_daily_data,
            # Averaged per day
            "59408-5": calculate_average_data,
            "61006-3": calculate_average_data,
            "9279-1": calculate_average_data,
            "40443-4": calculate_average_data,
            "HKQuantityTypeIdentifierWalkingHeartRateAverage": calculate_average_data,
        }

        # Example of ranges that could be used for filtering
        self.default_value_ranges = {
            "9052-2": (0, 2700),  # Calorie intake total (kcal)
            "55423-8": (0, 30000),  # Step count (steps)
            # Averaged per day
            "8867-4": (34, 200),  # Heart Rate (bpm)
            "80404-7": (40, 120),  # Heart Rate Variability SDNN (ms)
            "59408-5": (80, 100),  # Oxygen Saturation (%)
            "61006-3": (0.2, 5),  # Peripheral Perfusion Index (%)
            "9279-1": (10, 24),  # Respiratory Rate (breaths per minute)
            "40443-4": (40, 110),  # Resting Heart Rate (bpm)
            "HKQuantityTypeIdentifierWalkingHeartRateAverage": (
                55,
                120,
            ),  # Walking Heart Rate Average (bpm)
            # Neither averaged nor summed up
            "8462-4": (30, 100),  # Blood Pressure Diastolic (mmHg)
            "8480-6": (50, 200),  # Blood Pressure Systolic (mmHg)
            "8310-5": (34, 42),  # Body Temperature (C)
            "HKQuantityTypeIdentifierBasalBodyTemperature": (
                34,
                42,
            ),  # Basal Body Temperature (C)
            "74859-0": (0, 5),  # Blood Alcohol Content (%)
            "41653-7": (18, 190),  # Blood Glucose (mg/dL)
            "29463-7": (0, 1000),  # Body Mass (lbs)
            "39156-5": (5, 40),  # Body Mass Index (kg/m^2)
            "41982-0": (5, 60),  # Body Fat Percentage (%)
            "HKQuantityTypeIdentifierElectrodermalActivity": (),  # Electrodermal Activity (siemens)
            # Environmental Audio Exposure (dB(SPL))
            "HKQuantityTypeIdentifierEnvironmentalAudioExposure": (),
            "20150-9": (1, 6),  # Forced Expiratory Volume1 (L)
            "19870-5": (),  # Forced Vital Capacity (L)
            # Headphone Audio Exposure (dB(SPL))
            "HKQuantityTypeIdentifierHeadphoneAudioExposure": (),
            "8302-2": (),  # Height (in)
            "HKQuantityTypeIdentifierInhalerUsage": (),  # Inhaler Usage (count)
            "91557-9": (),  # Lean Body Mass (lbs)
            "HKQuantityTypeIdentifierNumberOfTimesFallen": (),  # NumberOfTimesFallen (falls)
            "19935-6": (),  # Peak Expiratory Flow Rate (L/min)
            "HKQuantityTypeIdentifierPhysicalEffort": (),  # Physical Effort (kcal/hr/kg)
            "96502-0": (),  # Push Count (wheelchair pushes)
            "HKQuantityTypeIdentifierUVExposure": (),  # UV Exposure (count)
            "HKQuantityTypeIdentifierVO2Max": (),  # VO2Max (mL/kg/min)
            "8280-0": (),  # Waist Circumference (in)
        }
