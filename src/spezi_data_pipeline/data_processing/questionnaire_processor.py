#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
This module provides a collection of functions designed for the processing of questionnaire
responses represented in the FHIR (Fast Healthcare Interoperability Resources) format. It includes
capabilities for calculating risk scores associated with the following questionnaires:
- Walking Impairment Questionnaire (WIQ),
- PHQ-9

The functions are tailored to work with `FHIRDataFrame`, a custom data structure that encapsulates
FHIR data in a pandas DataFrame.
"""

# Standard library imports
from enum import Enum

# Related third-party imports
import pandas as pd

# Local application/library specific imports
from spezi_data_pipeline.data_flattening.fhir_resources_flattener import (
    FHIRDataFrame,
    ColumnNames,
)

OBJECT = "object"


class DepressionSeverity(Enum):
    """
    Enumeration representing different levels of depression severity.

    Attributes:
    NONE_MINIMAL (tuple): A tuple representing none to minimal depression with a score range of
                          0 to 4.
    MILD (tuple): A tuple representing mild depression with a score range of 5 to 9.
    MODERATE (tuple): A tuple representing moderate depression with a score range of 10 to 14.
    MODERATELY_SEVERE (tuple): A tuple representing moderately severe depression with a score
                               range of 15 to 19.
    SEVERE (tuple): A tuple representing severe depression with a score range of 20 to 27.
    INVALID (tuple): A tuple representing an invalid score.
    """

    NONE_MINIMAL = (0, 4, "None-minimal depression")
    MILD = (5, 9, "Mild depression")
    MODERATE = (10, 14, "Moderate depression")
    MODERATELY_SEVERE = (15, 19, "Moderately severe depression")
    SEVERE = (20, 27, "Severe depression")
    INVALID = (-1, -1, "Invalid score")

    def __init__(self, min_score, max_score, description):
        self.min_score = min_score
        self.max_score = max_score
        self.description = description

    @staticmethod
    def interpret_score(score):
        """
        Interpret the given score and return the corresponding depression severity description.

        Parameters:
        score (int): The score to interpret.

        Returns:
        str: The description of the corresponding depression severity.
        """
        for severity in DepressionSeverity:
            if severity.min_score <= score <= severity.max_score:
                return severity.description
        return DepressionSeverity.INVALID.description


def calculate_risk_score(fhir_dataframe):  # pylint: disable=unused-variable
    """
    Calculate the risk score for grouped rows and add score interpretation.

    Parameters:
    fhir_dataframe (FHIRDataFrame): The input dataframe with columns 'UserId', 'AuthoredDate',
                                   'SurveyTitle', and 'AnswerCode'.

    Returns:
    pd.DataFrame: The resulting dataframe with added 'RiskScore' and 'ScoreInterpretation' columns.
    """

    if fhir_dataframe.df[ColumnNames.ANSWER_CODE.value].dtype == OBJECT:
        fhir_dataframe.df[ColumnNames.ANSWER_CODE.value] = fhir_dataframe.df[
            ColumnNames.ANSWER_CODE.value
        ].astype(int)

    if fhir_dataframe.df[ColumnNames.AUTHORED_DATE.value].dtype == OBJECT:
        fhir_dataframe.df[ColumnNames.AUTHORED_DATE.value] = pd.to_datetime(
            fhir_dataframe.df[ColumnNames.AUTHORED_DATE.value]
        )

    grouped_df = fhir_dataframe.df.groupby(
        [
            # ColumnNames.USER_ID.value,
            ColumnNames.AUTHORED_DATE.value,
            ColumnNames.SURVEY_TITLE.value,
        ],
        as_index=False,
    )[ColumnNames.ANSWER_CODE.value].sum()
    grouped_df.rename(
        columns={ColumnNames.ANSWER_CODE.value: "RiskScore"}, inplace=True
    )

    grouped_df["ScoreInterpretation"] = grouped_df["RiskScore"].apply(
        DepressionSeverity.interpret_score
    )

    return FHIRDataFrame(grouped_df, fhir_dataframe.resource_type)
