import pandas as pd
import numpy as np
import re                        
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt   
import seaborn as sns
import plotly.express as px

with zipfile.ZipFile("NYPD_Hate_Crimes.zip", "r") as z:
    with z.open("NYPD_Hate_Crimes.csv") as f:
        df_hate_crimes = pd.read_csv(f)
columns_to_drop = ['Complaint Precinct Code', 'Law Code Category Description',  'PD Code Description','Bias Motive Description','Month Number','Patrol Borough Name','Full Complaint ID']
df_hate_crimes = df_hate_crimes.drop(columns=[col for col in columns_to_drop if col in  df_hate_crimes.columns])
df_hate_crimes = df_hate_crimes.drop_duplicates()
df_hate_crimes = df_hate_crimes.dropna()
#print(df_hate_crimes.head())

with zipfile.ZipFile("NYPD_Arrest_Data__Year_to_Date_.zip", "r") as z:
    with z.open("NYPD_Arrest_Data__Year_to_Date_.csv") as f:
        df_arrest = pd.read_csv(f)
df_arrest.replace("(null)", np.nan, inplace=True)
#print("Columns in the Arrest DataFrame:")
#print(df_arrest.columns)
columns_to_drop = ['PD_CD', 'PD_DESC', 'KY_CD', 'LAW_CODE', 'LAW_CAT_CD','JURISDICTION_CODE', 'X_COORD_CD', 'Y_COORD_CD','Latitude', 'Longitude', 'New Georeferenced Column'
]
columns_to_drop = [col for col in columns_to_drop if col in df_arrest.columns]
#print(f"Columns to drop: {columns_to_drop}")
df_arrest.drop(columns=columns_to_drop, inplace=True)
df_arrest.dropna(inplace=True)
#print("Cleaned Arrest DataFrame:")
#print(df_arrest.head())


df_shooting = pd.read_csv("NYPD_shooting_incident_data__Historic__.csv")
df_shooting.replace("(null)", np.nan, inplace=True)
columns_to_keep = ['OCCUR_DATE', 'BORO', 'VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX', 'PERP_SEX', 'PERP_RACE', 'PRECINCT']
df_shooting = df_shooting[columns_to_keep]
df_shooting.dropna(inplace=True)
#print(df_shooting.head())


df_hate_crimes.rename(columns={'Arrest Date': 'DATE', 'County': 'BORO'}, inplace=True)
df_hate_crimes['DATE'] = pd.to_datetime(df_hate_crimes['DATE'])
df_arrest.rename(columns={'ARREST_DATE': 'DATE', 'ARREST_BORO': 'BORO'}, inplace=True)
df_arrest['DATE'] = pd.to_datetime(df_arrest['DATE'])
df_shooting.rename(columns={'OCCUR_DATE': 'DATE'}, inplace=True)
df_shooting['DATE'] = pd.to_datetime(df_shooting['DATE'])
merged_df = pd.merge(df_arrest, df_shooting, on=['DATE', 'BORO'], how='outer')
merged_df = pd.merge(merged_df, df_hate_crimes, on=['DATE', 'BORO'], how='outer')
#print(merged_df.head())
merged_df.to_csv("merged_nypd_data.csv", index=False)

def FinalDataset():
    print(merged_df.head(2))



############## THE FOLLOWING IS SAAD CODE FOR EDA ################## 

race_counts = merged_df['PERP_RACE_y'].value_counts()
#print(race_counts)
race_counts = merged_df['VIC_RACE'].value_counts()
#print(race_counts)
sex_counts = merged_df['PERP_SEX_y'].value_counts()
#print(sex_counts)
age_group_counts = merged_df['AGE_GROUP'].value_counts()
#print(age_group_counts)
age_group_counts = merged_df['AGE_GROUP'].value_counts()
#print(age_group_counts)
merged_df['BORO'] = merged_df['BORO'].replace({
    'K': 'BROOKLYN', 
    'M': 'MANHATTAN', 
    'B': 'BRONX', 
    'Q': 'QUEENS', 
    'S': 'STATEN ISLAND',
    'KINGS': 'BROOKLYN',
    'NEW YORK': 'MANHATTAN',
    'RICHMOND': 'STATEN ISLAND'})
boro_counts = merged_df['BORO'].value_counts()
#print(boro_counts)
offense_desc_counts = merged_df['OFNS_DESC'].value_counts()
#print(offense_desc_counts)
precinct_counts = merged_df['PRECINCT'].value_counts()
#print(precinct_counts)

def get_perp_race_counts(df):
    return df['PERP_RACE_y'].value_counts()

def get_vic_race_counts(df):
    return df['VIC_RACE'].value_counts()

def get_sex_counts(df):
    return df['PERP_SEX_y'].value_counts()

def get_age_group_counts(df):
    return df['AGE_GROUP'].value_counts()

def get_boro_counts(df):
    df['BORO'] = df['BORO'].replace({
        'K': 'BROOKLYN', 
        'M': 'MANHATTAN', 
        'B': 'BRONX', 
        'Q': 'QUEENS', 
        'S': 'STATEN ISLAND',
        'KINGS': 'BROOKLYN',
        'NEW YORK': 'MANHATTAN',
        'RICHMOND': 'STATEN ISLAND'})
    return df['BORO'].value_counts()

def get_offense_desc_counts(df):
    return df['OFNS_DESC'].value_counts()

def get_precinct_counts(df):
    return df['PRECINCT'].value_counts()
