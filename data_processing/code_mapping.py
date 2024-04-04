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
from .observation_processing import calculate_daily_data, calculate_average_data


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
        }

        # Maps LOINC codes and similar identifiers to processing functions
        self.code_to_function = {
            # Summed up per day
            "9052-2": calculate_daily_data,  # Calorie intake total (kcal)
            "55423-8": calculate_daily_data,  # Step count (steps)
            "HKQuantityTypeIdentifierDietaryProtein": calculate_daily_data,  # Dietary Protein (g)
            "41981-2": calculate_daily_data,  # Active Energy Burned (kcal)
            "HKQuantityTypeIdentifierAppleExerciseTime": calculate_daily_data,  # Apple Exercise Time (min)
            "HKQuantityTypeIdentifierAppleMoveTime": calculate_daily_data,  # Apple Move Time (min)
            "HKQuantityTypeIdentifierAppleStandTime": calculate_daily_data,  # Apple Stand Time (min)
            "HKQuantityTypeIdentifierBasalEnergyBurned": calculate_daily_data,  # Basal Energy Burned (kcal)
            "HKQuantityTypeIdentifierDietaryBiotin": calculate_daily_data,  # Dietary Biotin (ug)
            "HKQuantityTypeIdentifierDietaryCaffeine": calculate_daily_data,  # Dietary Caffeine (mg)
            "HKQuantityTypeIdentifierDietaryCalcium": calculate_daily_data,  # Dietary Calcium (mg)
            "HKQuantityTypeIdentifierDietaryCarbohydrates": calculate_daily_data,  # Dietary Carbohydrates (g)
            "HKQuantityTypeIdentifierDietaryChloride": calculate_daily_data,  # Dietary Chloride (mg)
            "HKQuantityTypeIdentifierDietaryChoresterol": calculate_daily_data,  # Dietary Choresterol (mg)
            "HKQuantityTypeIdentifierDietaryChromium": calculate_daily_data,  # Dietary Chromium (ug)
            "HKQuantityTypeIdentifierDietaryCopper": calculate_daily_data,  # Dietary Copper (ug)
            "HKQuantityTypeIdentifierDietaryFatMonounsaturated": calculate_daily_data,  # Dietary Fat Monounsaturated (g)
            "HKQuantityTypeIdentifierDietaryFatPolyunsaturated": calculate_daily_data,  # Dietary Fat Polyunsaturated (g)
            "HKQuantityTypeIdentifierDietaryFatSaturated": calculate_daily_data,  # Dietary Fat Saturated (g)
            "HKQuantityTypeIdentifierDietaryFat": calculate_daily_data,  # Dietary Fat Total (g)
            "HKQuantityTypeIdentifierDietaryFiber": calculate_daily_data,  # Dietary Fiber (g)
            "HKQuantityTypeIdentifierDietaryFolate": calculate_daily_data,  # Dietary Folate (ug)
            "HKQuantityTypeIdentifierDietaryIodine": calculate_daily_data,  # Dietary Iodine (ug)
            "HKQuantityTypeIdentifierDietaryIron": calculate_daily_data,  # Dietary Iron (mg)
            "HKQuantityTypeIdentifierDietaryMagnesium": calculate_daily_data,  # Dietary Magnesium (mg)
            "HKQuantityTypeIdentifierDietaryManganese": calculate_daily_data,  # DietaryManganese (mg)
            "HKQuantityTypeIdentifierDietaryMolybdenum": calculate_daily_data,  # Dietary Molybdenum (ug)
            "HKQuantityTypeIdentifierDietaryNiacin": calculate_daily_data,  # Dietary Niacin (mg)
            "HKQuantityTypeIdentifierDietaryPantothenicAcid": calculate_daily_data,  # Dietary Pantothenic Acid (mg)
            "HKQuantityTypeIdentifierDietaryPhosphorus": calculate_daily_data,  # Dietary Phosphorus (mg)
            "HKQuantityTypeIdentifierDietaryPotassium": calculate_daily_data,  # Dietary Potassium (mg)
            "HKQuantityTypeIdentifierDietaryProtein": calculate_daily_data,  # Dietary Protein (g)
            "HKQuantityTypeIdentifierDietaryRiboflavin": calculate_daily_data,  # Dietary Riboflavin (mg)
            "HKQuantityTypeIdentifierDietarySelenium": calculate_daily_data,  # Dietary Selenium (ug)
            "HKQuantityTypeIdentifierDietarySodium": calculate_daily_data,  # Dietary Sodium (mg)
            "HKQuantityTypeIdentifierDietarySugar": calculate_daily_data,  # Dietary Sugar (g)
            "HKQuantityTypeIdentifierDietaryThiamin": calculate_daily_data,  # Dietary Thiamin (mg)
            "HKQuantityTypeIdentifierDietaryVitaminA": calculate_daily_data,  # Dietary Vitamin A (ug)
            "HKQuantityTypeIdentifierDietaryVitaminB12": calculate_daily_data,  # Dietary Vitamin B12 (ug)
            "HKQuantityTypeIdentifierDietaryVitaminB6": calculate_daily_data,  # Dietary Vitamin B6 (mg)
            "HKQuantityTypeIdentifierDietaryVitaminC": calculate_daily_data,  # Dietary Vitamin C (mg)
            "HKQuantityTypeIdentifierDietaryVitaminD": calculate_daily_data,  # Dietary Vitamin D (ug)
            "HKQuantityTypeIdentifierDietaryVitaminE": calculate_daily_data,  # Dietary Vitamin E (mg)
            "HKQuantityTypeIdentifierDietaryVitaminK": calculate_daily_data,  # Dietary Vitamin K (ug)
            "HKQuantityTypeIdentifierDietaryWater": calculate_daily_data,  # Dietary Water (l)
            "HKQuantityTypeIdentifierDietaryZinc": calculate_daily_data,  # Dietary Zinc (mg)
            "9052-2": calculate_daily_data,  # Dietary Energy Consumed (kcal)
            "HKQuantityTypeIdentifierDistanceCycling": calculate_daily_data,  # Distance Cycling (m)
            "HKQuantityTypeIdentifierDistanceDownhillSnowSports": calculate_daily_data,  # Distance Downhill Snow Sports (m)
            "93816-7": calculate_daily_data,  # Distance Swimming (m)
            "HKQuantityTypeIdentifierDistanceWalkingRunning": calculate_daily_data,  # Distance Walking Running (m)
            "HKQuantityTypeIdentifierDistanceWheelchair": calculate_daily_data,  # Distance Wheelchair (m)
            "100304-5": calculate_daily_data,  # Flights Climbed (flights)
            "HKQuantityTypeIdentifierSwimmingStrokeCount": calculate_daily_data,  # Swimming Stroke Count (strokes)
            # Averaged per day
            "8867-4": calculate_average_data,  # Heart Rate (bpm)
            "80404-7": calculate_average_data,  # Heart Rate Variability SDNN (ms)
            "59408-5": calculate_average_data,  # Oxygen Saturation (%)
            "61006-3": calculate_average_data,  # Peripheral Perfusion Index (%)
            "9279-1": calculate_average_data,  # Respiratory Rate (breaths per minute)
            "40443-4": calculate_average_data,  # Resting Heart Rate (bpm)
            "HKQuantityTypeIdentifierWalkingHeartRateAverage": calculate_average_data,  # Walking Heart Rate Average (bpm)
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
            "HKQuantityTypeIdentifierEnvironmentalAudioExposure": (),  # Environmental Audio Exposure (dB(SPL))
            "20150-9": (1, 6),  # Forced Expiratory Volume1 (L)
            "19870-5": (),  # Forced Vital Capacity (L)
            "HKQuantityTypeIdentifierHeadphoneAudioExposure": (),  # Headphone Audio Exposure (dB(SPL))
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
