import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

def load_data():
    file = 'aircrahesFullDataUpdated_2024.csv'
    df = pd.read_csv(file)

    return df
st.markdown("""
    <style>
    /* Set background color */
    .reportview-container {
        background: #F0F0F5; /* Light white background */
    }

    /* Main title and subtitles color */
    .stApp {
        color: #4B0082; /* Dark Purple */
    }

    /* Headers (e.g., subheader) styling */
    h1, h2, h3, h4, h5, h6 {
        color: #4B0082 !important; /* Dark Purple */
    }

    /* Adjust the dataframes' text color and other elements */
    .stDataFrame {
        color: #4B0082; /* Purple */
    }

    /* Sidebar styling (if you have a sidebar) */
    .sidebar .sidebar-content {
        background: #4B0082 !important; /* Dark Purple */
        color: white; /* Text color */
    }

    /* Customizing buttons */
    button {
        background-color: #4B0082 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

df = load_data()
df.head()
# Step 1:  Handling Missing Values and Standardizing Entries

# Replace '-' with NaN in 'Country/Region'
df['Country/Region'] = df['Country/Region'].replace('-', pd.NA)

# Fill missing values in 'Country/Region' and 'Operator' with 'Unknown'
df['Country/Region'] = df['Country/Region'].fillna('Unknown')
df['Operator'] = df['Operator'].fillna('Unknown')
# Step 2: Trim and Clean Text Fields
# Columns to clean: 'Country/Region', 'Aircraft', 'Location', 'Operator'
df['Country/Region'] = df['Country/Region'].str.strip().str.title()
df['Aircraft Manufacturer'] = df['Aircraft Manufacturer'].str.strip().str.title()
df['Aircraft'] = df['Aircraft'].str.strip()
df['Location'] = df['Location'].str.strip()
df['Operator'] = df['Operator'].str.strip()
# Step 3: Convert 'Quarter' and 'Month' to categorical data types
df['Quarter'] = pd.Categorical(df['Quarter'], categories=['Qtr 1', 'Qtr 2', 'Qtr 3', 'Qtr 4'])
df['Month'] = pd.Categorical(df['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])
# Step 4: Handling Outliers
# Calculate the interquartile range (IQR) for 'Ground' and 'Fatalities (air)' to detect outliers
Q1_ground = df['Ground'].quantile(0.25)
Q3_ground = df['Ground'].quantile(0.75)
IQR_ground = Q3_ground - Q1_ground

Q1_fatalities = df['Fatalities (air)'].quantile(0.25)
Q3_fatalities = df['Fatalities (air)'].quantile(0.75)
IQR_fatalities = Q3_fatalities - Q1_fatalities

# Define outlier bounds
lower_bound_ground = Q1_ground - 1.5 * IQR_ground
upper_bound_ground = Q3_ground + 1.5 * IQR_ground

lower_bound_fatalities = Q1_fatalities - 1.5 * IQR_fatalities
upper_bound_fatalities = Q3_fatalities + 1.5 * IQR_fatalities

# Cap outliers in 'Ground' and 'Fatalities (air)'
df['Ground'] = df['Ground'].clip(lower=lower_bound_ground, upper=upper_bound_ground)
df['Fatalities (air)'] = df['Fatalities (air)'].clip(lower=lower_bound_fatalities, upper=upper_bound_fatalities)
# Step 5: Check for Duplicates
duplicate_rows = df.duplicated().sum()
# If any duplicates are found, remove them
df.drop_duplicates(inplace=True)
# Title of the app
st.title("Aircraft Crashes Data Explorer")

# Display the raw dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Research Question 1: Year with the highest number of air fatalities
st.subheader("1. Year with the Highest Number of Air Fatalities")
yearly_fatalities = df.groupby('Year')['Fatalities (air)'].sum().reset_index()
highest_fatality_year = yearly_fatalities.sort_values(by='Fatalities (air)', ascending=False).head(1)
st.write(highest_fatality_year)

# Research Question 2: Aircraft manufacturer with the highest number of accidents
st.subheader("2. Aircraft Manufacturer with the Most Accidents")
manufacturer_accidents = df['Aircraft Manufacturer'].value_counts().reset_index()
manufacturer_accidents.columns = ['Aircraft Manufacturer', 'Accident Count']
top_manufacturer = manufacturer_accidents.head(1)
st.write(top_manufacturer)

# Research Question 3: Aircraft manufacturers by total air fatalities (Graph)
st.subheader("3. Aircraft Manufacturers by Total Air Fatalities (Graph)")
manufacturer_fatalities = df.groupby('Aircraft Manufacturer')['Fatalities (air)'].sum().reset_index()
top_10_manufacturers = manufacturer_fatalities.sort_values(by='Fatalities (air)', ascending=False).head(10)

# Plot the top 10 manufacturers by air fatalities
plt.figure(figsize=(10, 6))
sns.barplot(x='Fatalities (air)', y='Aircraft Manufacturer', data=top_10_manufacturers)
plt.title('Top 10 Aircraft Manufacturers by Air Fatalities')
st.pyplot(plt)

# Research Question 4: Operators involved in the most crashes (Graph)
st.subheader("4. Operators Involved in the Most Crashes (Graph)")
operator_accidents = df['Operator'].value_counts().reset_index()
operator_accidents.columns = ['Operator', 'Accident Count']
top_10_operators = operator_accidents.head(10)

# Plot the top 10 operators by accident count
plt.figure(figsize=(10, 6))
sns.barplot(x='Accident Count', y='Operator', data=top_10_operators)
plt.title('Top 10 Operators by Number of Accidents')
st.pyplot(plt)

# Research Question 5: Yearly crash trends (Graph)
st.subheader("5. Yearly Crash Trends (Graph)")
yearly_crashes = df.groupby('Year').size().reset_index(name='Crash Count')

# Plot the trend of crashes over the years
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Crash Count', data=yearly_crashes)
plt.title('Trend of Total Crashes Over the Years')
plt.ylabel('Number of Crashes')
plt.xlabel('Year')
st.pyplot(plt)

# Option to explore more columns interactively with graphical representations
st.subheader("Explore Crashes by Selected Columns")

# Multi-select for column selection
selected_columns = st.multiselect("Select columns to filter crashes", ['Aircraft Manufacturer', 'Operator', 'Year'])

if selected_columns:
    # If 'Year' is selected, show it in tabular form with multiselect for specific years
    if 'Year' in selected_columns:
        st.subheader("Number of Crashes per Year (Table)")
        
        # Create a multiselect option for specific year selection and sort the years in ascending order
        sorted_years = sorted(df['Year'].unique())  # Sort years in ascending order
        selected_years = st.multiselect("Select specific years to display", sorted_years)

        # Filter the data by selected years
        if selected_years:
            year_crashes = df[df['Year'].isin(selected_years)].groupby('Year').size().reset_index(name='Crash Count')
        else:
            year_crashes = df.groupby('Year').size().reset_index(name='Crash Count')  # Default to all years if none selected
        
        st.write(year_crashes)
        selected_columns.remove('Year')  # Remove 'Year' from further graphical representation
    
    # For all other selected columns, show different chart types
    for col in selected_columns:
        column_crashes = df[col].value_counts().reset_index()
        column_crashes.columns = [col, 'Crash Count']

        # Dropdown to select chart type
        chart_type = st.selectbox(f"Choose chart type for {col}", ['Bar Chart', 'Line Chart', 'Pie Chart'])

        # Bar Chart
        if chart_type == 'Bar Chart':
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Crash Count', y=col, data=column_crashes.head(10))  # Limiting to top 10 for better visualization
            plt.title(f'Number of Crashes by {col}')
            st.pyplot(plt)

        # Line Chart
        elif chart_type == 'Line Chart':
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=col, y='Crash Count', data=column_crashes.head(10))  # Limiting to top 10
            plt.title(f'Trend of Crashes by {col}')
            st.pyplot(plt)

        # Pie Chart
        elif chart_type == 'Pie Chart':
            plt.figure(figsize=(10, 6))
            plt.pie(column_crashes['Crash Count'].head(10), labels=column_crashes[col].head(10), autopct='%1.1f%%', startangle=140)
            plt.title(f'Crash Distribution by {col}')
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            st.pyplot(plt)