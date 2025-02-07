<!--

This source file is part of the Stanford Spezi open-source project.

SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# Spezi Data Pipeline

[![Build and Test](https://github.com/StanfordSpezi/SpeziDataPipeline/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/StanfordSpezi/SpeziDataPipeline/actions/workflows/build-and-test.yml)
[![codecov](https://codecov.io/gh/StanfordSpezi/SpeziDataPipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/StanfordSpezi/SpeziDataPipeline)

The Spezi Data Pipeline offers a comprehensive suite of tools designed to facilitate the management, analysis, and visualization of healthcare data from Firebase Firestore. By adhering to the Fast Healthcare Interoperability Resources (FHIR) standards, this platform ensures that data handling remains robust, standardized, and interoperable across different systems and software.

## Overview

The Spezi Data Pipeline is engineered to improve workflows associated with data accessibility and analysis in healthcare environments. It supports [82 HKQuantityTypes](https://github.com/StanfordBDHG/HealthKitOnFHIR/blob/main/Sources/HealthKitOnFHIR/HealthKitOnFHIR.docc/SupportedHKQuantityTypes.md) and ECG data recordings and is capable of performing functions such as selection, storage, downloading, basic filtering, statistical analysis, and graphical representation of data. By facilitating the structured export of data from Firebase and incorporating FHIR standards, the pipeline enhances interoperability and streamlines data operations.

## Package Structure

The Spezi Data Pipeline is organized into several directories, each serving a specific function as part of the overall application. This guide will walk you through the package structure, highlighting the key components and their usage based on your needs and challenges.

1. `data_access/`

_FirebaseFHIRAccess_
- Purpose: Connects to a Firebase Firestore database and fetches the data stored as FHIR resources.
- Usage: If you need to retrieve healthcare data from a Firestore database, this class provides methods to connect to the database and fetch data based on LOINC codes.

_ResourceCreator_
- Purpose: Creates FHIR resource objects from Firestore documents in FHIR format.
- Usage: Use this when you need to convert raw FHIR-compatible Firestore documents into structured FHIR resources.

2. `data_flattening/`

_ResourceFlattener_
- Purpose: Transforms nested FHIR resources objects into flat data structures suitable for analysis.
- Usage: Essential for converting complex FHIR resources into a more analyzable DataFrame format.

3. `data_processing/`

_FHIRDataProcessor_
- Purpose: Processes and filters flattened FHIR data.
- Usage: Ideal for performing operations like filtering outliers, selecting data by user or date, averaging data by date, and general data processing tasks.

_CodeProcessor_
- Purpose: Handles processing related to code mappings.
- Usage: Use this when you need to map codes to meaningful representations. This class serves as a central repository for the mappings of LOINC codes to their display names, processing functions, and default value ranges for outlier filtering.

4. `data_exploration/`

_DataExplorer_
- Purpose: Provides tools for visualizing and exploring FHIR data.
- Usage: Useful for generating plots and visual representations of your data to gain insights, and detect user inactivity and missing values.

_ECGExplorer_
- Purpose: Specialized in visualizing ECG data.
- Usage: Use this for detailed ECG data analysis and visualization.

5. `data_export/`

_DataExporter_
- Purpose: Exports processed and visualized data to various formats.
- Usage: When you need to save your processed data or visualizations, this class provides methods to export to CSV and save plots in JPEG/PNG.


### How to Use Based on Your Needs
- **Downloading Data from Firestore**: Start with `FirebaseFHIRAccess` to connect and fetch data.
- **Converting and Structuring FHIR Data**: Use `ResourceCreator` and its subclasses to convert Firestore documents to FHIR resources.
- **Flattening Nested FHIR Data**: Utilize `ResourceFlattener` and its specific implementations to transform data into flat `DataFrames`.
- **Processing Data**: Apply FHIRDataProcessor for filtering, selecting, and general data processing tasks.
- **Exploring and Visualizing Data**: Leverage `DataExplorer` and `ECGExplorer`, and `QuestionnaireResponseExplorer` to create visualizations and explore your data.
- **Exporting Data**: Use `DataExporter` to save processed data and plots.


## Dependencies

Required Python packages are included in the requirements.txt file and are outlined in the list below:

**[pandas](https://pypi.org/project/pandas/)**

**[numpy](https://numpy.org/doc/stable/user/install.html)**

**[matplotlib](https://pypi.org/project/matplotlib/)**

**[firebase_admin](https://firebase.google.com/docs/admin/setup)**

**[fhir.resources](https://pypi.org/project/fhir.resources/)**

You can install all required external packages using pip by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Generate Service Account Key

To interact with Firebase services like Firestore or the Realtime Database, ensure your Firebase project is configured correctly and possesses the necessary credentials file (usually a .JSON file).

Visit the "Project settings" in your Firebase project, navigate to the "Service accounts" tab, and generate a new private key by clicking on "Generate new private key." Upon confirmation, the key will be downloaded to your system.

This .JSON file contains your service account credentials and is used to authenticate your application with Firebase.

## Usage Example

### Configuration

```python
# Path to the Firebase service account key file
serviceAccountKey_file = "path/to/your/serviceAccountKey.json"

# Firebase project ID
project_id = "projectId"

# Collection details within Firebase Firestore. Replace with the collection names in your project.
collection_name = "users"
subcollection_name = "HealthKit"

```

> [!NOTE]
>
> - Replace "path/to/your/serviceAccountKey.json" with the actual path to the .JSON file you downloaded earlier.
> - The "projectId" is your Firebase project ID, which you can find in your Firebase project settings.

### Connect to Firebase

```python

# Initialize and connect to Firebase using FHIR standards
firebase_access = FirebaseFHIRAccess(project_id, service_account_key_file)
firebase_access.connect()
```

## Observations

### Data Handling

In this example, we will demonstrate how we can perform Firestore query to download step counts (LOINC code: 55423-8) and heart rate (LOINC code: 8867-4) data, and, subsequently, to flatten them in a more readable and convenient tabular format.

```python
# Select the LOINC codes for the HealthKit quantities to perform a Firebase query
loinc_codes = ["55423-8", "8867-4"]

# Fetch and flatten FHIR data
fhir_observations = firebase_access.fetch_data(collection_name, subcollection_name, loinc_codes)
flattened_fhir_dataframe = flatten_fhir_resources(fhir_observations)
```
> [!NOTE]
>
> - If loinc_codes are omitted from the input arguments, FHIR resources for all stored LOINC codes are downloaded.


### Using Full Path to Fetch Data

You can alternatively specify the full path to a collection in Firestore to fetch data. This ensures that you can use Spezi Data Pipeline flatteners on any location within your database.

```python
# Fetch FHIR data from full path
collection_path = "collection/subcollection/......"
fhir_observations = firebase_access.fetch_data_path(collection_path, loinc_codes)
```

### Firebase Index Date Filtering

If you want to fetch data from a particular date range, you can setup a Firebase index within [Firestore](https://firebase.google.com/docs/firestore/query-data/indexing). You can then specify a `start_date` and `end_date` to filter your data before it is streamed. This can help prevent fetching unnecessary data. 

```python
# Sort FHIR data by date and fetch from full path
collection_path = "collection/subcollection/......"
start_date = "2024-04-22"
end_date = "2024-04-24"
fhir_observations = firebase_access.fetch_data_path(collection_path, loinc_codes, start_date, end_date)
flattened_fhir_dataframe = flatten_fhir_resources(fhir_observations)
```

### Apply basic processing for convenient data readability

Spezi Data Pipeline offers basic functions for improved data organization and readability. For example, individual step count data instances can be grouped by date using the process_fhir_data() function. If no intuitive function needs to be performed, the data remain unchanged.

```python
processed_fhir_dataframe = FHIRDataProcessor().process_fhir_data(flattened_fhir_dataframe)
```

### Create visual representations to explore the data

The dowloaded data can be then plotted using the following commands:

```python
# Create a visualizer instance
visualizer = DataVisualizer()

# Set plotting configuration
selected_users = ["User1","User2", "User3"]
selected_start_date = "2022-03-01"
selected_end_date = "2024-03-13"

# Select users and dates to plot
visualizer.set_user_ids(selected_users)
visualizer.set_date_range(selected_start_date, selected_end_date)

# Generate the plot
figs = visualizer.create_static_plot(processed_fhir_dataframe)
```

![daily_steps_data_plot.png](https://raw.githubusercontent.com/StanfordSpezi/SpeziDataPipeline/main/Figures/daily_steps_data_plot.png)
![heart_rate_data_plot.png](https://raw.githubusercontent.com/StanfordSpezi/SpeziDataPipeline/main/Figures/heart_rate_data_plot.png)


## ECG Observations

In a similar way, we can download and flatten ECG recordings (LOINC code: 131329) that are stored in Firestore.

### Create visual representations to explore the data

```python
# Create a visualizer instance
visualizer = ECGVisualizer()

# Set plotting configuration
selected_users = ["User1"]

selected_start_date = "2023-03-13"
selected_end_date = "2023-03-13"

# Select users and dates to plot
visualizer.set_user_ids(selected_users)
visualizer.set_date_range(selected_start_date, selected_end_date)

# Generate the plot
figs = visualizer.plot_ecg_subplots(processed_fhir_dataframe)
```

![ecg_data_plot.png](https://raw.githubusercontent.com/StanfordSpezi/SpeziDataPipeline/main/Figures/ecg_data_plot.png)


### Questionnaire Responses
The Spezi Data Pipeline also handles questionnaire responses stored as FHIR resources, facilitating the collection and analysis of questionnaire data in a standardized format. In addition, it includes calculation formulas for risk scores for certain questionnaire types based on the provided questionnaire responses.

> [!NOTE]
> 
> In FHIR standards, the `Questionnaire` resource represents the definition of a questionnaire, including questions and possible answers, while the `QuestionnaireResponse` resource captures the responses to a completed questionnaire, containing the answers provided by a user or patient.


## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/StanfordSpezi/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/StanfordSpezi/.github/blob/main/CODE_OF_CONDUCT.md) first.

## License

This project is licensed under the MIT License. See [Licenses](https://github.com/StanfordSpezi/SpeziAccessGuard/tree/main/LICENSES) for more information.

![Spezi Footer](https://raw.githubusercontent.com/StanfordSpezi/.github/main/assets/FooterLight.png#gh-light-mode-only)
![Spezi Footer](https://raw.githubusercontent.com/StanfordSpezi/.github/main/assets/FooterDark.png#gh-dark-mode-only)
