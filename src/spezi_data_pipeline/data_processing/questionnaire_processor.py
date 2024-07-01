#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Module for calculating risk scores based on various health questionnaires using FHIR resources.

This module defines enumerations for different levels of severity for depression, anxiety, 
and impairment. It includes functions to calculate risk scores for specific questionnaires 
like PHQ-9, GAD-7, and WIQ, using the FHIRDataFrame structure from the Spezi data pipeline.

Classes:
    DepressionSeverity(Enum): Enumeration representing different levels of depression severity
                              for PHQ-9.
    AnxietySeverity(Enum): Enumeration representing different levels of anxiety severity for GAD-7.
    ImpairmentSeverity(Enum): Enumeration representing different levels of impairment for WIQ.
    SupportedQuestionnaires(Enum): Enumeration representing supported questionnaires.

Functions:
    calculate_aggregated_score(fhir_dataframe, severity_enum): Calculate the risk score for
        questionnaires with similar logic to PHQ-9 and GAD-7.
    calculate_wiq_score(fhir_dataframe): Calculate the risk score for the WIQ questionnaire.
    calculate_risk_score(fhir_dataframe, questionnaire_title): Calculate the risk score based
        on the questionnaire title.

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
    Enumeration representing different levels of depression severity for PHQ-9.

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


class AnxietySeverity(Enum):
    """
    Enumeration representing different levels of anxiety severity for GAD-7.

    Attributes:
    MINIMAL (tuple): A tuple representing minimal anxiety with a score range of 0 to 4.
    MILD (tuple): A tuple representing mild anxiety with a score range of 5 to 9.
    MODERATE (tuple): A tuple representing moderate anxiety with a score range of 10 to 14.
    SEVERE (tuple): A tuple representing severe anxiety with a score range of 15 to 21.
    INVALID (tuple): A tuple representing an invalid score.
    """

    MINIMAL = (0, 4, "Minimal anxiety")
    MILD = (5, 9, "Mild anxiety")
    MODERATE = (10, 14, "Moderate anxiety")
    SEVERE = (15, 21, "Severe anxiety")
    INVALID = (-1, -1, "Invalid score")

    def __init__(self, min_score, max_score, description):
        self.min_score = min_score
        self.max_score = max_score
        self.description = description

    @staticmethod
    def interpret_score(score):
        """
        Interpret the given score and return the corresponding anxiety severity description.

        Parameters:
        score (int): The score to interpret.

        Returns:
        str: The description of the corresponding anxiety severity.
        """
        for severity in AnxietySeverity:
            if severity.min_score <= score <= severity.max_score:
                return severity.description
        return AnxietySeverity.INVALID.description


class ImpairmentSeverity(Enum):
    """
    Enumeration representing different levels of impairment for WIQ.

    Attributes:
    HIGH_IMPAIRMENT (tuple): A tuple representing high impairment with a score range of 0 to 10.
    MODERATE_IMPAIRMENT (tuple): A tuple representing moderate impairment with a score range of
                                 20 to 50.
    LOW_IMPAIRMENT (tuple): A tuple representing low impairment with a score range of 60 to 80.
    NO_IMPAIRMENT (tuple): A tuple representing no impairment with a score range of 90 to 100.
    INVALID (tuple): A tuple representing an invalid score.
    """

    HIGH_IMPAIRMENT = (0, 10, "High impairment")
    MODERATE_IMPAIRMENT = (20, 50, "Moderate impairment")
    LOW_IMPAIRMENT = (60, 80, "Low impairment")
    NO_IMPAIRMENT = (90, 100, "No impairment")
    INVALID = (-1, -1, "Invalid score")

    def __init__(self, min_score, max_score, description):
        self.min_score = min_score
        self.max_score = max_score
        self.description = description

    @staticmethod
    def interpret_score(score):
        """
        Interpret the given score and return the corresponding impairment description.

        Parameters:
        score (int): The score to interpret.

        Returns:
        str: The description of the corresponding impairment severity.
        """
        for severity in ImpairmentSeverity:
            if severity.min_score <= score <= severity.max_score:
                return severity.description
        return ImpairmentSeverity.INVALID.description


class SupportedQuestionnaires(Enum):
    """
    Enumeration representing supported questionnaires.

    Attributes:
    PHQ_9: The PHQ-9 questionnaire.
    GAD_7: The GAD-7 questionnaire.
    WIQ: The WIQ questionnaire.
    """

    PHQ_9 = "PHQ-9"
    GAD_7 = "GAD-7"
    WIQ = "WIQ"


def calculate_aggregated_score(
    fhir_dataframe: FHIRDataFrame, severity_enum: Enum
) -> FHIRDataFrame:
    """
    Calculate the risk score for questionnaires with similar logic to PHQ-9 and GAD-7.

    Parameters:
    fhir_dataframe (FHIRDataFrame): The input dataframe with columns 'UserId', 'AuthoredDate',
                                   'SurveyTitle', and 'AnswerCode'.
    severity_enum (Enum): The enumeration class to interpret the score.

    Returns:
    FHIRDataFrame: The resulting dataframe with added 'RiskScore' and 'ScoreInterpretation'
                   columns, with rows corresponding to grouped_df and other columns populated
                   with N/A.
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
            ColumnNames.AUTHORED_DATE.value,
            ColumnNames.SURVEY_TITLE.value,
        ],
        as_index=False,
    )[ColumnNames.ANSWER_CODE.value].sum()

    grouped_df.rename(
        columns={ColumnNames.ANSWER_CODE.value: ColumnNames.RISK_SCORE.value},
        inplace=True,
    )

    grouped_df[ColumnNames.SCORE_INTERPRETATION.value] = grouped_df[
        ColumnNames.RISK_SCORE.value
    ].apply(severity_enum.interpret_score)

    result_df = pd.DataFrame(columns=fhir_dataframe.df.columns)
    result_df[ColumnNames.RISK_SCORE.value] = pd.NA
    result_df[ColumnNames.SCORE_INTERPRETATION.value] = pd.NA

    # Populate the result DataFrame with rows from grouped_df
    new_rows = []
    for _, row in grouped_df.iterrows():
        new_row = {col: pd.NA for col in result_df.columns}
        new_row.update(
            {
                ColumnNames.AUTHORED_DATE.value: row[ColumnNames.AUTHORED_DATE.value],
                ColumnNames.SURVEY_TITLE.value: row[ColumnNames.SURVEY_TITLE.value],
                ColumnNames.RISK_SCORE.value: row[ColumnNames.RISK_SCORE.value],
                ColumnNames.SCORE_INTERPRETATION.value: row[
                    ColumnNames.SCORE_INTERPRETATION.value
                ],
            }
        )
        new_rows.append(new_row)

    result_df = pd.concat([result_df, pd.DataFrame(new_rows)], ignore_index=True)

    return FHIRDataFrame(result_df, resource_type=fhir_dataframe.resource_type)


def calculate_wiq_score(
    fhir_dataframe: FHIRDataFrame, severity_enum: Enum
) -> FHIRDataFrame:
    """
    Calculate the risk score for the WIQ questionnaire.

    Parameters:
    fhir_dataframe (FHIRDataFrame): The input dataframe with columns 'UserId', 'AuthoredDate',
                                   'SurveyTitle', and 'AnswerCode'.
    severity_enum (Enum): The enumeration class to interpret the score.

    Returns:
    FHIRDataFrame: The resulting dataframe with added 'RiskScore' and 'ScoreInterpretation' columns.
    """

    # Mapping from distance to impairment percentage
    wiq_mapping = {
        50: 0,
        150: 10,
        300: 20,
        600: 50,
        900: 80,
        1500: 100,
    }

    if fhir_dataframe.df[ColumnNames.ANSWER_CODE.value].dtype == OBJECT:
        fhir_dataframe.df[ColumnNames.ANSWER_CODE.value] = fhir_dataframe.df[
            ColumnNames.ANSWER_CODE.value
        ].astype(int)

    if fhir_dataframe.df[ColumnNames.AUTHORED_DATE.value].dtype == OBJECT:
        fhir_dataframe.df[ColumnNames.AUTHORED_DATE.value] = pd.to_datetime(
            fhir_dataframe.df[ColumnNames.AUTHORED_DATE.value]
        )

    fhir_dataframe.df["ImpairmentScore"] = fhir_dataframe.df[
        ColumnNames.ANSWER_CODE.value
    ].map(wiq_mapping)

    grouped_df = fhir_dataframe.df.groupby(
        [
            ColumnNames.AUTHORED_DATE.value,
            ColumnNames.SURVEY_TITLE.value,
        ],
        as_index=False,
    )["ImpairmentScore"].mean()

    grouped_df.rename(
        columns={"ImpairmentScore": ColumnNames.RISK_SCORE.value}, inplace=True
    )

    grouped_df[ColumnNames.SCORE_INTERPRETATION.value] = grouped_df[
        ColumnNames.RISK_SCORE.value
    ].apply(severity_enum.interpret_score)

    result_df = pd.DataFrame(columns=fhir_dataframe.df.columns)
    result_df[ColumnNames.RISK_SCORE.value] = pd.NA
    result_df[ColumnNames.SCORE_INTERPRETATION.value] = pd.NA

    # Populate the result DataFrame with rows from grouped_df
    new_rows = []
    for _, row in grouped_df.iterrows():
        new_row = {col: pd.NA for col in result_df.columns}
        new_row.update(
            {
                ColumnNames.AUTHORED_DATE.value: row[ColumnNames.AUTHORED_DATE.value],
                ColumnNames.SURVEY_TITLE.value: row[ColumnNames.SURVEY_TITLE.value],
                ColumnNames.RISK_SCORE.value: row[ColumnNames.RISK_SCORE.value],
                ColumnNames.SCORE_INTERPRETATION.value: row[
                    ColumnNames.SCORE_INTERPRETATION.value
                ],
            }
        )
        new_rows.append(new_row)

    result_df = pd.concat([result_df, pd.DataFrame(new_rows)], ignore_index=True)

    return FHIRDataFrame(result_df, resource_type=fhir_dataframe.resource_type)


def calculate_risk_score(  # pylint: disable=unused-variable
    fhir_dataframe: FHIRDataFrame, questionnaire_title: str
) -> FHIRDataFrame:
    """
    Calculate the risk score based on the questionnaire title.

    Parameters:
    fhir_dataframe (pd.DataFrame): The input dataframe with columns 'UserId', 'AuthoredDate',
                                   'SurveyTitle', and 'AnswerCode'.
    questionnaire_title (str): The title of the questionnaire to determine the calculation logic.

    Returns:
    FHIRDataFrame: The resulting dataframe with added 'RiskScore' and 'ScoreInterpretation' columns
                  or other appropriate structure based on the questionnaire.
    """

    calculation_functions = {
        SupportedQuestionnaires.PHQ_9.value: (
            calculate_aggregated_score,
            DepressionSeverity,
        ),
        SupportedQuestionnaires.GAD_7.value: (
            calculate_aggregated_score,
            AnxietySeverity,
        ),
        SupportedQuestionnaires.WIQ.value: (calculate_wiq_score, ImpairmentSeverity),
        # Add other questionnaires here
    }

    if questionnaire_title in calculation_functions:
        func, severity_enum = calculation_functions[questionnaire_title]
        if severity_enum:
            return func(fhir_dataframe, severity_enum)

    available_options = ", ".join(calculation_functions.keys())
    raise ValueError(
        f"Unsupported questionnaire title: {questionnaire_title}. \
                        Available options: {available_options}"
    )
