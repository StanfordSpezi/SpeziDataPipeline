#
# This source file is part of the Stanford Spezi open-source project
#
# SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import pandas as pd
import numpy as np
import csv
import uuid
import ast
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from re import sub
import glob
import json
import pytz
import ipywidgets as widgets
from IPython.display import display
from pandas import to_datetime

def vizualize_data(dfs_dict):
    dfs_dict = rename_key_dfs(dfs_dict)

    plot_output = widgets.Output()
    slider_lower = widgets.IntSlider(value=0, min=0, max=1000, step=1, description='Min value:', continuous_update=False)
    slider_upper = widgets.IntSlider(value=780, min=0, max=5000, step=1, description='Max value:', continuous_update=False)
    date_picker_start = widgets.DatePicker(description='Start Date:', disabled=False)
    date_picker_end = widgets.DatePicker(description='End Date:', disabled=False)

    def update_user_dropdown(df_key):
        unique_users = dfs_dict[df_key]['UserId'].unique().tolist()
        dropdown_user.options = unique_users
        return unique_users

    def update_plot():
        with plot_output:
            plot_output.clear_output(wait=True)
            selected_df = dfs_dict[dropdown_df.value]
            selected_user = dropdown_user.value
            start_date = date_picker_start.value
            end_date = date_picker_end.value
            y_lower = slider_lower.value
            y_upper = slider_upper.value
            
            if y_lower == y_upper:
                print("Error: Lower and upper threshold values must be different.")
                return
            
            str_start_date = start_date.strftime("%Y-%m-%d") if start_date else None
            str_end_date = end_date.strftime("%Y-%m-%d") if end_date else None
            
            if str_start_date and str_end_date:
                filtered_df = selected_df[(selected_df['EffectiveDateTime'] >= str_start_date) & (selected_df['EffectiveDateTime'] <= str_end_date)]
            else:
                filtered_df = selected_df

            filtered_df = filtered_df[(filtered_df['QuantityValue'] >= y_lower) & (filtered_df['QuantityValue'] <= y_upper)]

            if selected_user:
                filtered_df = filtered_df[filtered_df['UserId'] == selected_user]

            if filtered_df.empty:
                print("No data available to plot for the selected criteria.")
                return 

            if not selected_df.empty:
                plot_data(plot_output, filtered_df, str_start_date, str_end_date, user_id=selected_user, y_lower=y_lower, y_upper=y_upper, same_plot=True, save_as_tif=False)
              
 
            
    def on_dropdown_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            if change['owner'] == dropdown_df:
                update_user_dropdown(change['new'])
            update_plot()

    dropdown_df = widgets.Dropdown(options=list(dfs_dict.keys()), description='Data type:')
    dropdown_user = widgets.Dropdown(description='User:')

    dropdown_df.observe(on_dropdown_change)
    dropdown_user.observe(on_dropdown_change)
    slider_lower.observe(lambda change: update_plot(), names='value')
    slider_upper.observe(lambda change: update_plot(), names='value')
    date_picker_start.observe(lambda change: update_plot(), names='value')
    date_picker_end.observe(lambda change: update_plot(), names='value')

    display(dropdown_df, dropdown_user, slider_lower, slider_upper, date_picker_start, date_picker_end, plot_output)

    # Initialization
    if dropdown_df.options:
        update_user_dropdown(dropdown_df.options[0])
        dropdown_df.value = dropdown_df.options[0]
        
def analyze_data(reference_db, collection_name, date1=None, date2=None, save_as_csv=True):
    flattened_dfs = {}
    filtered_dfs = {}
    daily_dfs = {}
    
    unique_loinc_codes, display_to_loinc_dict = get_unique_codes_and_displays(reference_db, collection_name)

    for code in unique_loinc_codes:
        flattened_df = fetch_and_flatten_data(reference_db, 'users', code, save_as_csv)
        filtered_df = remove_outliers(flattened_df)
        daily_df = calculate_daily_data(filtered_df, save_as_csv)
    
        flattened_dfs[f'flattened_df_{code}'] = flattened_df
        filtered_dfs[f'filtered_df_{code}'] = filtered_df
        daily_dfs[f'daily_df_{code}'] = daily_df

    return flattened_dfs, filtered_dfs, daily_dfs


def get_unique_codes_and_displays(reference_db, collection_name):
    unique_loinc_codes = set()
    display_to_loinc_dict = {}  

    users = reference_db.collection(collection_name).stream()

    for user in users:
        healthkit_docs = reference_db.collection(collection_name).document(user.id).collection('HealthKit').stream()
        
        for doc in healthkit_docs:
            doc_data = doc.to_dict()
            coding = doc_data.get('code', {}).get('coding', [])
            if len(coding) > 0:
                loinc_code = coding[0]['code']
                # display = coding[0].get('display', '')
                quantity_name = coding[1].get('display', '') if len(coding) > 1 else (coding[0].get('display', '') if len(coding) > 0 else '')
                unique_loinc_codes.add(loinc_code)
                
                if quantity_name in display_to_loinc_dict:
                    display_to_loinc_dict[quantity_name].add(loinc_code)
                else:
                    display_to_loinc_dict[quantity_name] = {loinc_code}

    # Convert sets to lists in display_to_loinc_dict to make it JSON serializable
    for quantity_name in display_to_loinc_dict:
        display_to_loinc_dict[quantity_name] = list(display_to_loinc_dict[quantity_name])

    return unique_loinc_codes, display_to_loinc_dict

    

def rename_key_dfs(dfs_dict_or_df):
    if isinstance(dfs_dict_or_df, dict):
        renamed_dict_dfs = {}
        for key, df in dfs_dict_or_df.items():
            if not df.empty and 'QuantityName' in df.columns:
                new_key = df['QuantityName'].iloc[0]
                renamed_dict_dfs[new_key] = df
        return renamed_dict_dfs
    elif isinstance(dfs_dict_or_df, pd.DataFrame):
        if not dfs_dict_or_df.empty and 'QuantityName' in dfs_dict_or_df.columns:
            new_key = dfs_dict_or_df['QuantityName'].iloc[0]
            return {new_key: dfs_dict_or_df}
    else:
        raise ValueError("Input must be a DataFrame or a dictionary of DataFrames")

        


def fetch_and_flatten_data(reference_db, collection_name, input_code, save_as_csv=True):
    flattened_data = []
    users = reference_db.collection(collection_name).stream()
    
    for user in users:
        healthkit_ref = reference_db.collection(collection_name).document(user.id).collection('HealthKit')
        healthkit_docs = healthkit_ref.stream()
        
        for doc in healthkit_docs:
            doc_data = doc.to_dict()
            
            effective_datetime = doc_data.get('effectiveDateTime', doc_data.get('effectivePeriod', {}).get('start', ''))
            coding = doc_data.get('code', {}).get('coding', [])
            loinc_code = coding[0]['code'] if len(coding) > 0 else ''
            display = coding[0].get('display', '') if len(coding) > 0 else ''
            apple_healthkit_code = coding[1]['code'] if len(coding) > 1 else (coding[0]['code'] if len(coding) > 0 else '')
            quantity_name = coding[1].get('display', '') if len(coding) > 1 else (coding[0].get('display', '') if len(coding) > 0 else '')

            if loinc_code == input_code:
                flattened_entry = {
                    'UserId': user.id,
                    'DocumentId': doc_data.get('id', ''),
                    'EffectiveDateTime': effective_datetime,
                    'QuantityName': quantity_name,
                    'QuantityUnit': doc_data.get('valueQuantity', {}).get('unit', ''),
                    'QuantityValue': doc_data.get('valueQuantity', {}).get('value', ''),
                    'LoincCode': loinc_code,
                    'Display': display,
                    'AppleHealthKitCode': apple_healthkit_code,
                }

                flattened_data.append(flattened_entry)

    flattened_data = pd.DataFrame(flattened_data)
    print(flattened_data.columns)
    flattened_data['EffectiveDateTime'] = pd.to_datetime(flattened_data['EffectiveDateTime'], errors='coerce')

    if save_as_csv:
        filename = f'flattened_data_{snake_case(quantity_name)}_{datetime.now().strftime("%Y-%m-%d")}.csv'
        flattened_data.to_csv(filename, index=False)
        
    return flattened_data
  

def calculate_daily_data(df, save_as_csv=True):
    
    df['EffectiveDateTime'] = df['EffectiveDateTime'].dt.normalize()
    aggregated_data = df.groupby(['UserId', 'EffectiveDateTime'])['QuantityValue'].sum().reset_index()

    quantity_name = f"Daily {df['QuantityName'].iloc[0]}"
    unit = df['QuantityUnit'].iloc[0]
    display = f"Aggregated {df['QuantityName'].iloc[0]} data by date."
    code = df['LoincCode'].iloc[0]
    apple_code = df['AppleHealthKitCode'].iloc[0]

    aggregated_data['QuantityUnit'] = unit
    aggregated_data['QuantityName'] = quantity_name
    aggregated_data['Display'] = display
    aggregated_data['LoincCode'] = code
    aggregated_data['AppleHealthKitCode'] = apple_code

    if save_as_csv:
        filename = f'{snake_case(quantity_name)}_{datetime.now().strftime("%Y-%m-%d")}.csv'
        aggregated_data.to_csv(filename, index=False)

    return aggregated_data


def remove_outliers(df, manual_threshold=None):
    
    filtered_df = pd.DataFrame()
    
    default_thresholds = {'55423-8': 50, '9052-2': [1, 2500], 'HKQuantityTypeIdentifierDietaryProtein': [0, 600]}
        
    thresholds = manual_threshold or default_thresholds
    
    for code in df['LoincCode'].unique():
        threshold = thresholds.get(code, None)
        
        if threshold is not None:
            df_filtered_code = df[df['LoincCode'] == code]
            
            if isinstance(threshold, list):
                # Apply range conditions directly for '9052-2' and 'HKQuantityTypeIdentifierDietaryProtein'
                condition = (df_filtered_code['QuantityValue'] >= threshold[0]) & (df_filtered_code['QuantityValue'] <= threshold[1])
                df_to_keep = df_filtered_code[condition]
            else:
                # Special handling for '55423-8'
                aggregated_data = df_filtered_code.groupby(['DocumentId', 'EffectiveDateTime'])['QuantityValue'].sum().reset_index()
                condition = aggregated_data['QuantityValue'] > threshold
                days_to_keep = aggregated_data[condition][['DocumentId', 'EffectiveDateTime']]
                
                df_to_keep = pd.merge(df_filtered_code, days_to_keep, on=['DocumentId', 'EffectiveDateTime'], how='inner')
            
            filtered_df = pd.concat([filtered_df, df_to_keep], ignore_index=True)
        else:
            print(f"No threshold set for code {code}. Skipping.")
    
    if not filtered_df.empty:
        quantity_name = df['QuantityName'].iloc[0].lower().replace(' ', '_')
        filtered_df.to_csv(f'filtered_{quantity_name}_data_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv', index=False)
    
    return filtered_df

def print_statistics(df, user_id=None):
    if user_id is not None:
        df = df[df['UserId'] == user_id]
        users_to_process = [user_id]
    else:
        users_to_process = df['UserId'].unique()

    for user in users_to_process:
        user_df = df[df['UserId'] == user]
        
        if user_df.empty:
            print(f"No data available for user ID: {user}")
            continue

        # Assuming the LoincCode and QuantityUnit are consistent for each user
        description = user_df['Display'].iloc[0]
        unit = user_df['QuantityUnit'].iloc[0]
        
        daily_totals = user_df.groupby('EffectiveDateTime')['QuantityValue'].sum()

        mean_value = daily_totals.mean()
        std_dev_value = daily_totals.std()

        print(f"Statistics of {description.replace('.','')} for User ID: {user}:")
        print(f"Mean: {mean_value:.0f} {unit}")
        print(f"Standard Deviation: {std_dev_value:.0f} {unit}\n")    


def export_users_to_csv(reference_db, collection_name, csv_file_name):
    users = reference_db.collection(collection_name).stream()
    users_data = []
    all_identifiers = set()

    for user in users:
        user_data = user.to_dict()
        if user_data:
            user_data['User Document ID'] = user.id
            users_data.append(user_data)
            all_identifiers.update(user_data.keys())

    field_names = ['User Document ID'] + list(all_identifiers - {'User Document ID'})

    with open(csv_file_name, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()

        for user_data in users_data:
            csv_writer.writerow({field: user_data.get(field, '') for field in field_names})

    print(f'Data has been successfully written to {csv_file_name}.')
    print(f'The collection contains {len(users_data)} users.')
    return pd.read_csv(csv_file_name)


def print_data_summary(df):
    summary, user_count = summarize_user_entries(df)
    print(f"Total number of users: {user_count}")
    print("Number of entries per user:")
    print(summary.head())
    
def summarize_user_entries(df):
    entries_per_user = df.groupby('UserId').size().reset_index(name='Number of Entries')
    user_count = df['UserId'].nunique()
    
    return entries_per_user, user_count


def snake_case(s):
    return s.lower().replace(" ", "_")


def plot_data(plot_output, df, date1=None, date2=None, user_id=None, y_lower=None, y_upper=None, same_plot=True, save_as_tif=False):
    with plot_output:
        plot_output.clear_output(wait=True) 
        font_sizes = {
            'title': 14,
            'axes_label': 12,
            'tick_label': 10,
            'legend': 10
        }

        if date1 and date2:
            date1 = pd.to_datetime(datetime.strptime(date1, "%Y-%m-%d")).tz_localize('UTC')
            date2 = pd.to_datetime(datetime.strptime(date2, "%Y-%m-%d")).tz_localize('UTC')
            df = df[(df['EffectiveDateTime'] >= date1) & (df['EffectiveDateTime'] <= date2)]

        users_to_plot = [user_id] if user_id else df['UserId'].unique()

        def plot_user_data(user_df, user_id=None):
            title = f"{user_df['QuantityName'].iloc[0]}{' for All Dates' if not date1 and not date2 else f' from {date1} to {date2}'}{'' if same_plot else f' for User {user_id}'}"
            aggregated_data = user_df.groupby('EffectiveDateTime')['QuantityValue'].sum().reset_index()
            plt.bar(aggregated_data['EffectiveDateTime'].dt.normalize(), aggregated_data['QuantityValue'], color='skyblue', edgecolor='black', linewidth=1.5, label=f'User {user_id}' if same_plot else None)
            plt.ylim(y_lower, y_upper)
            
            if not same_plot:
                plt.title(title, fontsize=font_sizes['title'])
                plt.xlabel('Date', fontsize=font_sizes['axes_label'])
                plt.ylabel(f"{user_df['QuantityName'].iloc[0]} ({user_df['QuantityUnit'].iloc[0]})", fontsize=font_sizes['axes_label'])
                plt.xticks(rotation=45, fontsize=font_sizes['tick_label'])
                plt.yticks(fontsize=font_sizes['tick_label'])
                plt.tight_layout()
                if save_as_tif:
                    filename = f"{snake_case(user_df['QuantityName'].iloc[0])}_user_{user_id}_{'all_dates' if not date1 and not date2 else f'{date1}_to_{date2}'}.tif"
                    plt.savefig(filename, format='tif')
                plt.show()

        if same_plot:
            plt.figure(figsize=(10, 6))
            for uid in users_to_plot:
                user_df = df[df['UserId'] == uid]
                if not user_df.empty:
                    plot_user_data(user_df, uid)
            plt.ylim(y_lower, y_upper)
            plt.legend(fontsize=font_sizes['legend'])
            plt.xticks(rotation=45, fontsize=font_sizes['tick_label'])
            plt.yticks(fontsize=font_sizes['tick_label'])
            plt.title(f"{df['QuantityName'].iloc[0]}{' for All Users' if not user_id else f' for User {user_id}'}", fontsize=font_sizes['title'])
            plt.xlabel('Date', fontsize=font_sizes['axes_label'])
            plt.ylabel(f"{user_df['QuantityName'].iloc[0]} ({user_df['QuantityUnit'].iloc[0]})", fontsize=font_sizes['axes_label'])
            plt.tight_layout()
            if save_as_tif:
                filename = f"{snake_case(df['QuantityName'].iloc[0])}_{'all_dates' if not date1 and not date2 else f'{date1}_to_{date2}'}_{'all_users' if not user_id else f'user_{user_id}'}.tif"
                plt.savefig(filename, format='tif')
            plt.show()
            
        else:
            for uid in users_to_plot:
                user_df = df[df['UserId'] == uid]
                if not user_df.empty:
                    plt.figure(figsize=(10, 6))
                    plot_user_data(user_df, uid)
                    plt.show()
                    

def plot_and_export_data(df, date1=None, date2=None, user_id=None, y_lower=None, y_upper=None, same_plot=True, save_as_tif=False):
        font_sizes = {
            'title': 14,
            'axes_label': 12,
            'tick_label': 10,
            'legend': 10
        }
        
        if date1 and date2:
            date1 = pd.to_datetime(datetime.strptime(date1, "%Y-%m-%d")).tz_localize('UTC')
            date2 = pd.to_datetime(datetime.strptime(date2, "%Y-%m-%d")).tz_localize('UTC')
            df = df[(df['EffectiveDateTime'] >= date1) & (df['EffectiveDateTime'] <= date2)]

        users_to_plot = [user_id] if user_id else df['UserId'].unique()

        def plot_user_data(user_df, user_id=None):
            title = f"{user_df['QuantityName'].iloc[0]}{' for All Dates' if not date1 and not date2 else f' from {date1} to {date2}'}{'' if same_plot else f' for User {user_id}'}"
            aggregated_data = user_df.groupby('EffectiveDateTime')['QuantityValue'].sum().reset_index()
            plt.bar(aggregated_data['EffectiveDateTime'].dt.normalize(), aggregated_data['QuantityValue'], color='skyblue', edgecolor='black', linewidth=1.5, label=f'User {user_id}' if same_plot else None)
            plt.ylim(y_lower, y_upper)
            
            if not same_plot:
                plt.title(title, fontsize=font_sizes['title'])
                plt.xlabel('Date', fontsize=font_sizes['axes_label'])
                plt.ylabel(f"{user_df['QuantityName'].iloc[0]} ({user_df['QuantityUnit'].iloc[0]})", fontsize=font_sizes['axes_label'])
                plt.xticks(rotation=45, fontsize=font_sizes['tick_label'])
                plt.yticks(fontsize=font_sizes['tick_label'])
                plt.tight_layout()
                if save_as_tif:
                    filename = f"{snake_case(user_df['QuantityName'].iloc[0])}_user_{user_id}_{'all_dates' if not date1 and not date2 else f'{date1}_to_{date2}'}.tif"
                    plt.savefig(filename, format='tif')
                plt.show()

        if same_plot:
            plt.figure(figsize=(10, 6))
            for uid in users_to_plot:
                user_df = df[df['UserId'] == uid]
                if not user_df.empty:
                    plot_user_data(user_df, uid)
            plt.ylim(y_lower, y_upper)
            plt.legend(fontsize=font_sizes['legend'])
            plt.xticks(rotation=45, fontsize=font_sizes['tick_label'])
            plt.yticks(fontsize=font_sizes['tick_label'])
            plt.title(f"{df['QuantityName'].iloc[0]}{' for All Users' if not user_id else f' for User {user_id}'}", fontsize=font_sizes['title'])
            plt.xlabel('Date', fontsize=font_sizes['axes_label'])
            plt.ylabel(f"{user_df['QuantityName'].iloc[0]} ({user_df['QuantityUnit'].iloc[0]})", fontsize=font_sizes['axes_label'])
            plt.tight_layout()
            if save_as_tif:
                filename = f"{snake_case(df['QuantityName'].iloc[0])}_{'all_dates' if not date1 and not date2 else f'{date1}_to_{date2}'}_{'all_users' if not user_id else f'user_{user_id}'}.tif"
                plt.savefig(filename, format='tif')
            plt.show()
            
        else:
            for uid in users_to_plot:
                user_df = df[df['UserId'] == uid]
                if not user_df.empty:
                    plt.figure(figsize=(10, 6))
                    plot_user_data(user_df, uid)
                    plt.show()
