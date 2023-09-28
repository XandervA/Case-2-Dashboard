#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Importing libaries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import streamlit as st
import kaggle
from zipfile import ZipFile

# Necessary cause Ipython commands don't function after converting to a .py file
import subprocess

# Downloading dataset using kaggle API
command = 'kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020'
subprocess.run(command, shell=True)

# Unzipping file
file_name= "../Case-2-Dashboard/formula-1-world-championship-1950-2020.zip"
with ZipFile (file_name, 'r') as zip:
    zip.extractall()

# Assigning directory
data_dir = '../Case-2-Dashboard'

# Importing data
drivers_df_full = pd.read_csv(f'{data_dir}/drivers.csv')
qualifying_df_full = pd.read_csv(f'{data_dir}/qualifying.csv')
laptimes_df_full = pd.read_csv(f'{data_dir}/lap_times.csv')
constructor_standings_df_full = pd.read_csv(f'{data_dir}/constructor_standings.csv')
seasons_df_full = pd.read_csv(f'{data_dir}/seasons.csv')
circuits_df_full = pd.read_csv(f'{data_dir}/circuits.csv')
constructors_df_full = pd.read_csv(f'{data_dir}/constructors.csv')
constructors_results_df_full = pd.read_csv(f'{data_dir}/constructor_results.csv')
driver_standings_df_full = pd.read_csv(f'{data_dir}/driver_standings.csv')
results_df_full = pd.read_csv(f'{data_dir}/results.csv')
status_df_full = pd.read_csv(f'{data_dir}/status.csv')
races_df_full = pd.read_csv(f'{data_dir}/races.csv')



# Prepping data for merging 
races_df = races_df_full.copy()
races_df = races_df.drop(columns = ['time', 'url', 'fp1_date', 'fp1_time', 'fp2_date', 
                                    'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 
                                    'quali_time', 'sprint_date', 'sprint_time'])
races_df = races_df.rename(columns ={'name':'race_name'}) # Renaming name column to be more specific

constructor_standings_df = constructor_standings_df_full.copy()
constructor_standings_df = constructor_standings_df.drop(columns = ['constructorStandingsId'])

constructors_df = constructors_df_full.copy()
constructors_df = constructors_df.drop(columns = ['constructorRef', 'nationality', 'url'])
constructors_df = constructors_df.rename(columns = {'name' : 'constructors_name'}) # Renaming name column to be more specific

results_df = results_df_full.copy()
results_df = results_df.drop(columns = ['resultId', 'grid', 'position', 'positionText', 'positionOrder', 
                                        'laps', 'time', 'fastestLapTime', 'fastestLapSpeed', 'statusId'])
drivers_grouped = results_df.groupby(['raceId', 'constructorId'])['driverId'].agg(list).reset_index()

drivers_df = drivers_df_full.copy()
drivers_df = drivers_df.drop(columns = ['driverRef', 'number', 'code', 'forename', 'surname', 'url'])



# Defiding a custom aggregate function to add dob and nationality for all drivers connected to a given row in a single column
def custom_agg(series):
    return list(series.dropna())



# Merging dataframes
merged_df = pd.merge(constructors_df, constructor_standings_df, on = 'constructorId')
merged_df = merged_df.merge(races_df, on='raceId')
merged_df = merged_df.merge(drivers_grouped, on=['raceId', 'constructorId'], how='left')

merged_df = merged_df.explode('driverId')
merged_df = merged_df.merge(drivers_df, on='driverId', how='left')
merged_df = merged_df.groupby(['constructorId', 'constructors_name', 'raceId', 'points', 'position', 
                               'wins', 'year', 'round', 'circuitId', 'race_name', 'date'
                              ]).agg({'dob': custom_agg, 'nationality': custom_agg}).reset_index()




# Adding age column for later use
def calculate_age(row):
    date = pd.to_datetime(row['date'])
    dob_values = row['dob']
    ages = []
    for dob in dob_values:
        dob_date = pd.to_datetime(dob)
        age = (date - dob_date).days // 365
        ages.append(age)
    return ages

merged_df['age'] = merged_df.apply(calculate_age, axis=1)



# Lijsten om gegevens op te slaan
years = []
driver_ages = []

# Loop door elke rij in de oorspronkelijke DataFrame
for index, row in merged_df.iterrows():
    year = row['year']
    ages = row['age']
    
    # Voeg het jaar toe aan de lijst
    years.extend([year] * len(ages))
    
    # Voeg de leeftijden van de coureurs toe aan de lijst
    driver_ages.extend(ages)

# Maak een nieuwe DataFrame
data_ages_per_year = pd.DataFrame({'year': years, 'age': driver_ages})

# Sorteer de DataFrame op jaar (hoogste jaar bovenaan, laagste jaar onderaan)
data_ages_per_year = data_ages_per_year.sort_values(by='year', ascending=False)

# Bekijk het resultaat
data_ages_per_year.count()

st.title("How have Formula 1 drivers changed over the ages? (1950-2023)")
st.caption("By Emma Wartena, Luuk de Goede, Xander van Altena and Salah Bentaher")
st.caption("28th of September 2023")

st.write("Formula 1, the world’s fastest sport, has had many changes over the years. The most notable and remarkable changes are those made to the cars. However the cars are not the only thing that changed, the drivers have evolved too. This blogpost made for the Minor Data Science at Hogeschool van Amsterdam will go in depth about the changing faces of F1 Drivers over the decades. The changes discussed are based on data and backed up by visualizations. How the visualizations came about (code) will be displayed too. ")

st.image('https://publish.one37pm.net/wp-content/uploads/2022/07/F1RivalriesUniversal.png?fit=1600%2C707', caption='Credit: Alex Zito')

st.markdown("*The data used is extracted from the API Kaggle. The datasets have been made available by Rohan Rao and named Formula 1 World Championship 1950-2020 (named 2020 but goes up and till 2023). The data used is a combination of multiple datasets.*")

st.subheader("Driver’s gender")
st.write("Firstly, elephant in the room, the biggest change that could have occurred, but sadly didn’t really occur, is the gender of the drivers. Over the years there have only been 5 female drivers in Formula 1, of which two actually started a race. The last time this happened was in 1980 (source: the Guardian). However there have been multiple reports stating that there are several women who are coming up the ranks in racing. Hopefully someday we will see a lot more woman faces in the F1 cockpits. ")

st.subheader("Driver's age")
st.write("The following visualization shows the age deviation of all the drivers from each year between 1958 and 2023. All the driver’s ages pre-1958 weren’t recorded. By using the slider you can scroll through the years. ")

# Maak een lijst van unieke jaartallen
jaartallen = data_ages_per_year['year'].unique()

# Maak een subplot met een slider onder de figuur
fig = make_subplots(rows=2, cols=1, row_heights=[10, 1], specs=[[{'type': 'box'}], [{}]])
slider_steps = []

# Voeg elke jaarstap toe aan de slider
for jaar in jaartallen:
    slider_step = {
        'args': [
            {'y': [data_ages_per_year[data_ages_per_year['year'] == jaar]['age']]}
        ],
        'label': str(jaar),
        'method': 'restyle'
    }
    slider_steps.append(slider_step)

# Initialiseer de eerste zichtbare boxplot
initieel_jaar = jaartallen[0]
boxplot_data = [data_ages_per_year[data_ages_per_year['year'] == initieel_jaar]['age']]

# Voeg de eerste boxplot toe aan de subplot
fig.add_trace(go.Box(y=boxplot_data[0], boxmean=True, marker_color = 'indianred'))

# Voeg de slider toe onder de figuur
fig.update_layout(
    sliders=[{
        'steps': slider_steps,
        'active': 0,
        'x': 0.08,
        'y': 0.25,  # Plaats de slider onder de figuur
        'len': 0.85,
        'pad': {'t': 50}
    }],
    xaxis_title='Year',  # Pas de labels aan
    yaxis_title='Age',
    width=600, height=1000,
    
)

fig.update_traces(name='Age Deviation')
fig.update_yaxes(range=[10, 60], tickvals=list(range(15, 61, 5)))

# Toon de interactieve plot
st.plotly_chart(fig)
with st.expander("Code for Age Deviation plot"):
    code = '''# Maak een subplot met een slider onder de figuurfig = 
    make_subplots(rows=2, cols=1, row_heights=[10, 1], specs=[[{'type': 'box'}], [{}]])
    slider_steps = []

    # Voeg elke jaarstap toe aan de slider
    for jaar in jaartallen:
        slider_step = {
            'args': [
                {'y': [data_ages_per_year[data_ages_per_year['year'] == jaar]['age']]}
            ],
            'label': str(jaar),
            'method': 'restyle'
        }
        slider_steps.append(slider_step)

    # Initialiseer de eerste zichtbare boxplot
    initieel_jaar = jaartallen[0]
    boxplot_data = [data_ages_per_year[data_ages_per_year['year'] == initieel_jaar]['age']]

    # Voeg de eerste boxplot toe aan de subplot
    fig.add_trace(go.Box(y=boxplot_data[0], boxmean=True, marker_color = 'indianred'))

    # Voeg de slider toe onder de figuur
    fig.update_layout(
        sliders=[{
            'steps': slider_steps,
            'active': 0,
            'x': 0.08,
            'y': 0.25,  # Plaats de slider onder de figuur
            'len': 0.85,
            'pad': {'t': 50}
        }],
        xaxis_title='Year',  # Pas de labels aan
        yaxis_title='Age',
        width=600, height=1000,

    )

    fig.update_traces(name='Age Deviation')
    fig.update_yaxes(range=[10, 60], tickvals=list(range(15, 61, 5)))

    # Toon de interactieve plot
    st.plotly_chart(fig)"'''
    st.code(code, language="python", line_numbers=True)

st.write("Some notable changes have gone down over the years. The mean age has decreased by about 4 to 5 years. The conclusion can be made that the drivers as a collective have gotten slightly younger over the years. Another notable change is that drivers over the age of 45 are a thing of the past. ")

# List of final race IDs 
last_race = merged_df.groupby('year')['raceId'].max().reset_index()

# Filtering the dataframe to only display data for the last race of the season
merged_df2 = merged_df[merged_df.apply(lambda x: x['raceId'] == last_race[last_race['year'] == x['year']]
                                               ['raceId'].values[0], axis=1)]

final_df = merged_df2.sort_values(by=['year'], ascending=False)

# Filtering the results to only show constructors with > 5 seasons in F1
# Variable 'final_position_count > 10' can be altered to change the filter
final_df_count = final_df['constructorId'].value_counts()
final_df_count10 = final_df_count[final_df_count > 5].index
final_df_10 = final_df[final_df['constructorId'].isin(final_df_count10)]

st.subheader("Does age effect perforamnce?")
st.write("The following visualization shows the performances of some notable F1 teams with the mean age of the drivers competing for them in that year. You can select the teams with the dropdown menu.")

# Create a dropdown to select a constructor
constructor_dropdown = st.selectbox('Select Constructor:', final_df_10['constructors_name'].unique())

# Filter the dataframe based on the selected constructor
filtered_df = final_df_10[final_df_10['constructors_name'] == constructor_dropdown]

# Create a figure
fig2 = go.Figure()

# Add the trace for final position
fig2.add_trace(
    go.Scatter(
        x=filtered_df['year'],
        y=filtered_df['position'],
        mode='lines+markers',
        name='Final Position',
        marker=dict(symbol='circle', size=6, color='blue')
    )
)

# Calculate the mean age
mean_ages = []
for age_list in filtered_df['age']:
    age_list = [age for age in age_list if age != 0]
    if age_list:
        mean_age = sum(age_list) / len(age_list)
        mean_ages.append(mean_age)
    else:
        mean_ages.append(None)  # Set None for empty lists

# Add the trace for mean age
fig2.add_trace(
    go.Scatter(
        x=filtered_df['year'],
        y=mean_ages,
        mode='lines+markers',
        name='Mean Age',
        yaxis='y2',  # Use the second y-axis
        marker=dict(symbol='diamond', size=8, color='red')
    )
)

# Define the layout
fig2.update_layout(
    title=f'Constructor Performance vs. Mean Driver Age - {constructor_dropdown}',
    xaxis=dict(title='Year'),
    yaxis=dict(
        title='Final Position',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue'),
        autorange='reversed'
    ),
    yaxis2=dict(
        title='Mean Age',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right',  # Position the second y-axis on the right
    ),
    legend=dict(x=0.7, y=1.0),  # Adjust the legend position
)

# Display the Plotly chart using Streamlit
st.plotly_chart(fig2)
with st.expander("Code for Constructor Performance vs. Mean Driver Age plot"):
    code2 = '''st.header("Constructor Performance vs. Mean Driver Age")
    # Create a dropdown to select a constructor
    constructor_dropdown = st.selectbox('Select Constructor:', final_df_10['constructors_name'].unique())

    # Filter the dataframe based on the selected constructor
    filtered_df = final_df_10[final_df_10['constructors_name'] == constructor_dropdown]

    # Create a figure
    fig2 = go.Figure()

    # Add the trace for final position
    fig2.add_trace(
        go.Scatter(
            x=filtered_df['year'],
            y=filtered_df['position'],
            mode='lines+markers',
            name='Final Position',
            marker=dict(symbol='circle', size=6, color='blue')
        )
    )

    # Calculate the mean age
    mean_ages = []
    for age_list in filtered_df['age']:
        age_list = [age for age in age_list if age != 0]
        if age_list:
            mean_age = sum(age_list) / len(age_list)
            mean_ages.append(mean_age)
        else:
            mean_ages.append(None)  # Set None for empty lists

    # Add the trace for mean age
    fig2.add_trace(
        go.Scatter(
            x=filtered_df['year'],
            y=mean_ages,
            mode='lines+markers',
            name='Mean Age',
            yaxis='y2',  # Use the second y-axis
            marker=dict(symbol='diamond', size=8, color='red')
        )
    )

    # Define the layout
    fig2.update_layout(
        title=f'Constructor Performance vs. Mean Driver Age - {constructor_dropdown}',
        xaxis=dict(title='Year'),
        yaxis=dict(
            title='Final Position',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            autorange='reversed'
        ),
        yaxis2=dict(
            title='Mean Age',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right',  # Position the second y-axis on the right
        ),
        legend=dict(x=0.7, y=1.0),  # Adjust the legend position
    )'''
    st.code(code2, language="python", line_numbers=True)
    
st.write("The mean age line is an interesting line to analyze. In the older days of formula 1 this line is all over the places. In recent years there is a pattern visible. Usually teams appoint a younger driver, let him develop for a few years (the line increases by 1 every year) and then after a few years take up a younger driver again. The effect this might have on the teams performance can vary. Sometimes a team performs better and sometimes worse, usually the team doesn’t finish in the same place when new drivers are appointed. The conclusion you can take is that the ages of the teams drivers do effect the team’s performance. ")
    
# Vind de top 5 meest voorkomende nationaliteiten
top_nationalities = ['British', 'American', 'Italian', 'French', 'German']

st.subheader("Driver's nationality")
st.write("This visualization gives insight on drivers’ nationality over the years. The 5 most popular nationalities are filtered from the data and are highlighted. You can select and deselect the popular nationalities and other. The bars are stacked. ")

# Voeg het jaar van de race toe aan het results_df op basis van 'raceId'
results_df2 = results_df.merge(races_df[['raceId', 'year']], on='raceId')

# Voeg de nationaliteit van de coureurs toe aan het results_df op basis van 'driverId'
results_df2 = results_df2.merge(drivers_df[['driverId', 'nationality']], on='driverId')

# Voeg een nieuwe kolom 'nationality_group' toe met 'Overig' voor niet-topnationaliteiten
results_df2['nationality_group'] = (
    results_df2['nationality'].apply(lambda x: x if x in top_nationalities else 'Other')
)

# Bereken het aantal nationaliteiten per jaar
nationality_counts = (
    results_df2.groupby(['year', 'nationality_group'])['driverId']
             .nunique()
             .reset_index()
)


# Pivot de data voor eenvoudiger plotten
pivot_data = nationality_counts.pivot(index='year', columns='nationality_group', values='driverId').fillna(0)

colors = px.colors.qualitative.G10
# Plot het histogram met behulp van Plotly Express met kleuren
fig3 = px.bar(pivot_data, x=pivot_data.index, y=top_nationalities + ['Other'],
             labels={'year': 'Year', 'value': 'Count'},
             title='Different Nationalities',
             height=600,
             color_discrete_sequence=colors)

fig3.update_layout(barmode='stack')
st.plotly_chart(fig3)
with st.expander("Code for Different Nationalities plot"):
    code3 = '''# Vind de top 5 meest voorkomende nationaliteiten
    top_nationalities = ['British', 'American', 'Italian', 'French', 'German']

    st.header("Nationalities")
    # Voeg het jaar van de race toe aan het results_df op basis van 'raceId'
    results_df2 = results_df.merge(races_df[['raceId', 'year']], on='raceId')

    # Voeg de nationaliteit van de coureurs toe aan het results_df op basis van 'driverId'
    results_df2 = results_df2.merge(drivers_df[['driverId', 'nationality']], on='driverId')

    # Voeg een nieuwe kolom 'nationality_group' toe met 'Overig' voor niet-topnationaliteiten
    results_df2['nationality_group'] = (
        results_df2['nationality'].apply(lambda x: x if x in top_nationalities else 'Other')
    )

    # Bereken het aantal nationaliteiten per jaar
    nationality_counts = (
        results_df2.groupby(['year', 'nationality_group'])['driverId']
                 .nunique()
                 .reset_index()
    )


    # Pivot de data voor eenvoudiger plotten
    pivot_data = nationality_counts.pivot(index='year', columns='nationality_group', values='driverId').fillna(0)

    colors = px.colors.qualitative.G10
    # Plot het histogram met behulp van Plotly Express met kleuren
    fig3 = px.bar(pivot_data, x=pivot_data.index, y=top_nationalities + ['Other'],
                 labels={'year': 'Year', 'value': 'Count'},
                 title='Different Nationalities',
                 height=600,
                 color_discrete_sequence=colors)

    fig3.update_layout(barmode='stack')
    st.plotly_chart(fig3)'''
    st.code(code3, language="python", line_numbers=True)
st.write("The difference in count between the old years and recent years is significant. The reason for this is that the teams switched between drivers a lot more back in the day. In the recent days it’s more common to stick to the two drivers that you started the season with. It’s also noticeable that Americans have almost disappeared from the sport. In the past the contribution of drivers from the 5 most popular nationalities was very big. In the present time this contribution has gotten less. ")
st.subheader("TL-DR")
st.markdown("""
From the analyzed data, the following can be said about how Formula 1 drivers have changed over the years:

* Over the years, Formula 1 drivers have gotten 4 to 5 years younger on average.

* The age of a driver does affect the performance of a team. Teams' performance isn’t the same when a drastic change is made in the team’s drivers’ mean age.

* The nationalities of the drivers have shifted from mostly consisting of the 5 most popular nationalities to a more diverse set of nationalities.
""")


# In[ ]:




