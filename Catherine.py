#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import re                        
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt   
import seaborn as sns
import plotly.express as px


# In[34]:


df_shooting = pd.read_csv("NYPD_shooting_incident_data__Historic__.csv")
df_shooting.replace("(null)", np.nan, inplace=True)
columns_to_keep = ['OCCUR_DATE', 'BORO', 'VIC_RACE', 'VIC_AGE_GROUP', 'VIC_SEX', 'PERP_SEX', 'PERP_RACE', 'PRECINCT']
df_shooting = df_shooting[columns_to_keep]
df_shooting.dropna(inplace=True)
print(df_shooting.head())


# In[36]:


df_hate_crimes.rename(columns={'Arrest Date': 'DATE', 'County': 'BORO'}, inplace=True)
df_hate_crimes['DATE'] = pd.to_datetime(df_hate_crimes['DATE'])

df_arrest.rename(columns={'ARREST_DATE': 'DATE', 'ARREST_BORO': 'BORO'}, inplace=True)
df_arrest['DATE'] = pd.to_datetime(df_arrest['DATE'])

df_shooting.rename(columns={'OCCUR_DATE': 'DATE'}, inplace=True)
df_shooting['DATE'] = pd.to_datetime(df_shooting['DATE'])
 
merged_df = pd.merge(df_arrest, df_shooting, on=['DATE', 'BORO'], how='outer')
merged_df = pd.merge(merged_df, df_hate_crimes, on=['DATE', 'BORO'], how='outer')
 
print(merged_df.head())
#merged_df.to_csv("merged_nypd_data.csv", index=False)


# In[37]:


race_counts = merged_df['PERP_RACE_y'].value_counts()
print(race_counts)
race_counts = merged_df['VIC_RACE'].value_counts()
print(race_counts)
sex_counts = merged_df['PERP_SEX_y'].value_counts()
print(sex_counts)
age_group_counts = merged_df['AGE_GROUP'].value_counts()
print(age_group_counts)
age_group_counts = merged_df['AGE_GROUP'].value_counts()
print(age_group_counts)
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
print(boro_counts)
offense_desc_counts = merged_df['OFNS_DESC'].value_counts()
print(offense_desc_counts)
precinct_counts = merged_df['PRECINCT'].value_counts()
print(precinct_counts)

 


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
merged_df = pd.read_csv("merged_nypd_data.csv")
 
offense_boro = merged_df[['Offense Category', 'BORO']].dropna()
 
boro_mapping = {
    'KINGS': 'Brooklyn',
    'RICHMOND': 'Staten Island'
}
offense_boro['BORO'] = offense_boro['BORO'].replace(boro_mapping)

# top 5 offenses 
top_offenses = offense_boro['Offense Category'].value_counts().head(5).index
offense_boro = offense_boro[offense_boro['Offense Category'].isin(top_offenses)]

 
offense_boro_counts = offense_boro.groupby(['Offense Category', 'BORO']).size().reset_index(name='Crime Count')

 
plt.figure(figsize=(10, 5))
sns.barplot(data=offense_boro_counts, x='Offense Category', y='Crime Count', hue='BORO', palette='Set2')
plt.title('Crime Type Distribution Across NYC Boroughs')
plt.xlabel('Offense Category')
plt.ylabel('Number of Arrests')
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




