<!--

This source file is part of the Stanford Spezi open-source project.

SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
  
-->

# Spezi Data Pipeline Template

<a target="_blank" href="https://colab.research.google.com/github/StanfordSpezi/SpeziDataPipelineTemplate/blob/main/SpeziDataPipelineTemplate.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The Spezi Data Pipeline provides a suite of functionalities for accessing, filtering, analyzing, and visualizing data from Firestore, tailored specifically for healthcare applications. Designed for flexibility, it can be easily integrated into any Python notebook environment.

The `SpeziDataPipelineTemplate` Python Notebook contains a template demonstrating the usage of the Spezi Data Pipeline.


## Overview

The Spezi Data Pipeline is a Python-based tool designed to enhance data accessibility and analysis workflows, specifically tailored for healthcare applications. It suppors [82 HKQuantityTypes](https://github.com/StanfordBDHG/HealthKitOnFHIR/blob/main/Sources/HealthKitOnFHIR/HealthKitOnFHIR.docc/SupportedHKQuantityTypes.md), alowing users to select, save and download data, apply basic filtering methods, perform statistical analyses, and create visual data representations. Spezi Data Pipeline streamlines the process of exporting data from Firebase, facilitating structured decomposition, and provides a suite of out-of-the-box functionalities for data analysis.


### Dependencies

Ensure you have the following Python packages installed:

**[pandas](https://pypi.org/project/pandas/)** 

**[numpy](https://numpy.org/doc/stable/user/install.html)** 

**[matplotlib](https://matplotlib.org)** 

**[firebase_admin](https://firebase.google.com/docs/admin/setup)** 

You can install all required external packages using pip, which is the package installer for Python, with the following command:

```bash
pip install pandas numpy matplotlib firebase-admin
```


## Generate Service Account Key

When using firebase_admin for interactions with Firebase services, such as Firestore or the Realtime Database, make sure you have set up your Firebase project and have the necessary credentials file (usually a .json file) that you will use to initialize your application with firebase_admin.

In the "Project settings" page, navigate to the "Service accounts" tab. Here, you will find a section to generate a new private key for your Firebase project. Click on "Generate new private key" and confirm the action. A .json file will be downloaded to your computer.

This .json file contains your service account credentials and is used to authenticate your application with Firebase.



## Example

### 1. Initialize the Firebase admin SDK with your project credentials.
```python
# Path to the Firebase service account key file
serviceAccountKey_file = 'path/to/your/serviceAccountKey.json'

# Firebase project ID
project_id = 'projectId'

# Firestore collection name to be used
collection_name = 'users'

# Check if the Firebase app has already been initialized to avoid re-initialization
if not firebase_admin._apps:
    # Create a credential object from the service account key file
    cred = credentials.Certificate(serviceAccountKey_file)
    # Initialize the Firebase app with the credentials
    firebase_admin.initialize_app(cred)

# Create a Firestore client instance to interact with the database
db = firestore.client()
```


> [!NOTE]
> - Replace 'path/to/your/serviceAccountKey.json' with the actual path to the .json file you downloaded earlier.
> - The 'projectId' is your Firebase project ID, which you can find in your Firebase project settings.


### 2. Fetch and flatten data to retrieve and prepare your data.

Here, we present an example using the fetch_and_flatten() function to query data from Firestore for HKQuantityTypeIdentifierStepCount with LOINC code equal to 55423-8.

```python
flattened_df = fetch_and_flatten_data(db, collection_name, '55423-8')   
```

### 3. Apply remove_outliers to clean your dataset.

```python
filtered_df = remove_outliers(flattened_df)       
```

### 4. Aggregate data by date for each user to calculate daily data points.

```python
daily_df = calculate_daily_data(filtered_df)
```

### 5. Create visual representations to explore the data.

```python
vizualize_data(daily_df)

```
![visualize_data.png](attachment:b0a17b62-3676-4abb-bcb5-643f623c96ce.png)



## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/StanfordSpezi/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/StanfordSpezi/.github/blob/main/CODE_OF_CONDUCT.md) first.


## License

This project is licensed under the MIT License. See [Licenses](https://github.com/StanfordSpezi/SpeziAccessGuard/tree/main/LICENSES) for more information.

![Spezi Footer](https://raw.githubusercontent.com/StanfordSpezi/.github/main/assets/FooterLight.png#gh-light-mode-only)
![Spezi Footer](https://raw.githubusercontent.com/StanfordSpezi/.github/main/assets/FooterDark.png#gh-dark-mode-only)
