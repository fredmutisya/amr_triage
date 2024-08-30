import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import joblib
import plotly.graph_objects as go

# Import other necessary modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# For animation
import json
from streamlit_lottie import st_lottie

# Set page configuration
st.set_page_config(
    page_title="AST Prioritization Tool",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar expanded by default
)

# Load the Lottie animation
with open("doctor.json") as f:
    lottie_animation = json.load(f)

# Display the Lottie animation in the sidebar
with st.sidebar:
    st.sidebar.title("Instructions")
    st.sidebar.info("""
    This is an Antibiotic resistance triage tool developed by Dr. Oscar Nyangiri, Dr. Primrose Beryl, and Dr. Fred Mutisya as part of the Vivli Data challenge 2024. It makes use of the Pfizer Atlas data and the Venatorx Gears data. Input the patient details into the AST triage tool to assess the urgency of antimicrobial resistance testing.
    """)
    st_lottie(lottie_animation, height=300)

# Load saved model
model = pickle.load(open("decision_tree_model.pkl", 'rb'))

# Load the Variables CSV file for input options
Variables = pd.read_csv("Variables.csv")
Variables = Variables.fillna('')

# Load the main dataset from the provided CSV file
main_data_file = "combined_gears_atlas.csv"
combined_data = pd.read_csv(main_data_file)
combined_data = combined_data.fillna('')

# Load the World Bank countries with borders CSV file
world_bank_data = pd.read_csv("World_bank_countries_with_borders.csv", encoding='ISO-8859-1')

# Extract predictor names from the Variables CSV file
countries = Variables["Country"].tolist()
source_sample = Variables["Source"].tolist()
antibiotics = Variables["Antibiotics"].tolist()
Speciality = Variables["Speciality"].tolist()

# Set up the tabs
tab1, tab2 = st.tabs(["AST Triage Tool", "Performance of Decision trees in AST"])










with tab1:
    # AST Prioritization Tool Interface
    st.title('Antimicrobial Susceptibility Testing (AST) Triage Tool')
    st.markdown("""
    <div style="background-color:#ADD8E6;padding:10px">
    <h2 style="color:white;text-align:center;">Please answer the following questions about the patient:</h2>
    </div>
    """, unsafe_allow_html=True)

    # New patient-related input fields as questions
    age = st.selectbox('What is the age of the patient?', ['0 to 2 Years', '3 to 12 Years', '13 to 18 Years', '19 to 64 Years', '65 to 84 Years', '85 and Over'])
    gender = st.selectbox('What is the gender of the patient?', ['Male', 'Female', 'Other'])
    speciality = st.selectbox('What is the speciality of the patient?', Speciality)
    country = st.selectbox('In which country does the patient reside?', countries)
    history_resistance = st.selectbox('Does the patient have a known history of infection/colonization with resistant pathogens?', ['Yes', 'No'])
    hospitalization_history = st.selectbox('Has the patient been hospitalized, had day-clinic visits, or been in a care facility in the last 6 months?', ['Yes', 'No'])
    immunocompromised = st.selectbox('Does the patient have a recent or current history of malignancies, immunocompromise, or use of immunosuppressive drugs?', ['Yes', 'No'])
    recent_surgery = st.selectbox('Has the patient undergone surgery recently (in the past 3 months)?', ['Yes', 'No'])
    currently_hospitalized = st.selectbox('Is the patient currently hospitalized?', ['Yes', 'No'])
    icu = st.selectbox('Is the patient in the intensive care unit (ICU)?', ['Yes', 'No'])
    recent_antibiotics = st.selectbox('Has the patient taken antibiotics in the previous 3 months?', ['Yes', 'No'])
    suspected_infection = st.text_input('What type of infection do you suspect the patient has?')
    source = st.selectbox('What type of sample is to be collected?', source_sample)
    antibiotic = st.selectbox('Which antibiotic are you thinking of prescribing today?', antibiotics)

    # Filter the combined dataset based on the selected criteria
    filtered_data = combined_data[
        (combined_data['Country'] == country) &
        (combined_data['Source'] == source) 
        #(combined_data['Antibiotics'] == antibiotic)
    ]

    # Relax the criteria if no exact match is found
    if filtered_data.empty:
        # Check for bordering countries
        bordering_countries = world_bank_data.loc[world_bank_data['Country'] == country, 'Bordering Countries'].values
        if bordering_countries.size > 0 and isinstance(bordering_countries[0], str):
            bordering_countries_list = bordering_countries[0].split(', ')
            filtered_data = combined_data[
                (combined_data['Country'].isin(bordering_countries_list)) &
                (combined_data['Source'] == source) 
                #(combined_data['Antibiotics'] == antibiotic)
            ]
            # If a match is found, use it
            if not filtered_data.empty:
                st.write(f"Data found using bordering countries: {bordering_countries_list}")

        # If no data found with bordering countries, check the region
        if filtered_data.empty:
            region = world_bank_data.loc[world_bank_data['Country'] == country, 'Region Countries'].values
            if region.size > 0:
                region_countries = world_bank_data.loc[world_bank_data['Region Countries'] == region[0], 'Country'].tolist()
                filtered_data = combined_data[
                    (combined_data['Country'].isin(region_countries)) &
                    (combined_data['Source'] == source) 
                    #(combined_data['Antibiotics'] == antibiotic)
                ]
                # If a match is found, use it
                if not filtered_data.empty:
                    st.write(f"Data found using region countries: {region_countries}")

    # Ensure the 'Species', 'Antibiotics', and 'Resistance' columns are strings and handle NaN values
    filtered_data.loc[:, 'Species'] = filtered_data['Species'].astype(str).fillna('')
    #filtered_data.loc[:, 'Antibiotics'] = filtered_data['Antibiotics'].astype(str).fillna('')
    filtered_data.loc[:, 'Resistance'] = filtered_data['Resistance'].astype(str).fillna('')

    # Display buttons in two columns
    col1, col2 = st.columns(2)






    
    with col1:
        if st.button('Antibiogram'):
            if not filtered_data.empty:
                # Ensure that 'Species', 'Antibiotics', and 'Resistance' columns are correctly typed
                filtered_data.loc[:, 'Species'] = filtered_data['Species'].astype(str).fillna('')
                filtered_data.loc[:, 'Antibiotics'] = filtered_data['Antibiotics'].astype(str).fillna('')
                filtered_data.loc[:, 'Resistance'] = pd.to_numeric(filtered_data['Resistance'], errors='coerce').fillna(0).astype(int)
    
                # Calculate resistance counts by Species and Antibiotic
                resistance_summary = filtered_data.groupby(['Species', 'Antibiotics', 'Resistance']).size().unstack(fill_value=0)
    
                # Check if the '0' (Susceptible) column exists
                if 0 in resistance_summary.columns:
                    resistance_summary['Total Count'] = resistance_summary.sum(axis=1)
                    resistance_summary['% Susceptibility'] = (resistance_summary[0] / resistance_summary['Total Count']) * 100
                    resistance_summary = resistance_summary.round({'% Susceptibility': 1})
                    
                    # Format the percentage with a percentage sign
                    resistance_summary['% Susceptibility'] = resistance_summary['% Susceptibility'].apply(lambda x: f"{x:.1f}%")
                else:
                    st.write("No susceptible data available to calculate '% Susceptibility'.")
    
                # Reshape the DataFrame so that each antibiotic has its own column
                pivoted_susceptibility = resistance_summary.pivot_table(index='Species', columns='Antibiotics', values='% Susceptibility', aggfunc='first')
    
                # Ensure all antibiotics appear, even those not in the filtered data
                all_antibiotics = combined_data['Antibiotics'].unique()
                pivoted_susceptibility = pivoted_susceptibility.reindex(columns=all_antibiotics, fill_value='N/A')
    
                # Get the total count of each species (across all antibiotics) and insert it as the first column
                total_count = resistance_summary.groupby('Species')['Total Count'].first()
                pivoted_susceptibility.insert(0, 'Total Count', total_count)
    
                # Function to apply color formatting
                def color_format(val):
                    if isinstance(val, str) and val.endswith('%'):
                        percentage = float(val.rstrip('%'))
                        if percentage > 75:
                            color = 'green'
                        elif 50 <= percentage <= 75:
                            color = 'orange'
                        else:
                            color = 'red'
                        return f'background-color: {color}'
                    return ''
    
                # Apply the color formatting to the dataframe
                styled_summary = pivoted_susceptibility.style.applymap(color_format, subset=pd.IndexSlice[:, pivoted_susceptibility.columns[1:]])
    
                st.write("Detailed Antibiogram")
                st.dataframe(styled_summary)
            else:
                # If no data found, expand the selection to bordering countries
                bordering_countries = world_bank_data.loc[world_bank_data['Country'] == country, 'Bordering Countries'].values
                if bordering_countries.size > 0 and isinstance(bordering_countries[0], str):
                    bordering_countries_list = bordering_countries[0].split(', ')
                    filtered_data = combined_data.loc[
                        (combined_data['Country'].isin(bordering_countries_list)) &
                        (combined_data['Source'] == source)
                    ]
                    
                    if not filtered_data.empty:
                        st.write(f"No data for {country}, using bordering countries: {bordering_countries_list}")
                        # Re-run the antibiogram code with expanded selection
                        filtered_data.loc[:, 'Species'] = filtered_data['Species'].astype(str).fillna('')
                        filtered_data.loc[:, 'Antibiotics'] = filtered_data['Antibiotics'].astype(str).fillna('')
                        filtered_data.loc[:, 'Resistance'] = pd.to_numeric(filtered_data['Resistance'], errors='coerce').fillna(0).astype(int)
    
                        resistance_summary = filtered_data.groupby(['Species', 'Antibiotics', 'Resistance']).size().unstack(fill_value=0)
    
                        if 0 in resistance_summary.columns:
                            resistance_summary['Total Count'] = resistance_summary.sum(axis=1)
                            resistance_summary['% Susceptibility'] = (resistance_summary[0] / resistance_summary['Total Count']) * 100
                            resistance_summary = resistance_summary.round({'% Susceptibility': 1})
                            resistance_summary['% Susceptibility'] = resistance_summary['% Susceptibility'].apply(lambda x: f"{x:.1f}%")
                        else:
                            st.write("No susceptible data available to calculate '% Susceptibility'.")
    
                        pivoted_susceptibility = resistance_summary.pivot_table(index='Species', columns='Antibiotics', values='% Susceptibility', aggfunc='first')
                        pivoted_susceptibility = pivoted_susceptibility.reindex(columns=all_antibiotics, fill_value='N/A')
                        total_count = resistance_summary.groupby('Species')['Total Count'].first()
                        pivoted_susceptibility.insert(0, 'Total Count', total_count)
    
                        styled_summary = pivoted_susceptibility.style.applymap(color_format, subset=pd.IndexSlice[:, pivoted_susceptibility.columns[1:]])
    
                        st.write("Detailed Antibiogram")
                        st.dataframe(styled_summary)
                    else:
                        st.write(f"No data available for {country} or its bordering countries.")
                else:
                    st.write(f"No bordering countries data available for {country}.")


    
    



    
    # Button to show AI triaging result
    with col2:
        if st.button('AI Triaging Result'):
            susceptibility = ''
            if not filtered_data.empty:
                # Create a DataFrame with the input data for the model
                example_data = pd.DataFrame({
                    'Age.Group': [age],
                    'Country': [country],
                    'Speciality': [speciality], 
                    'Source': [source],
                    'Antibiotics': [antibiotic]
                })

                # Encode the input data (assuming the model requires label encoding)
                for column in example_data.columns:
                    le = LabelEncoder()
                    example_data[column] = le.fit_transform(example_data[column].astype(str))

                # Ensure the DataFrame aligns with the trained model
                encoded_values = pd.get_dummies(example_data)

                # Align encoded DataFrame with the model's expected feature names
                encoded_values = encoded_values.reindex(columns=model.feature_names_in_, fill_value=0)

                # Predict with the model
                prediction = model.predict_proba(encoded_values)[0]

                # Interpret the prediction
                susceptibility_percentage = prediction[1] * 100  # Probability of resistance
                if susceptibility_percentage >= 50:
                    susceptibility = f'Strong AST recommendation due to the high probability of resistance in similar cases from surveillance datasets.'
                else:
                    susceptibility = f'Moderate AST recommendation due to the lower probability of resistance in similar cases from surveillance datasets.'

                # Display the result
                st.write('Prediction:', susceptibility)
                st.write("""
                Disclaimer: The predictive AI model provided is intended for informational purposes only.
                """)


















with tab2:
    st.title("Performance of the Decision Tree model")
    st.write('This is a tool for policy makers to test the performance metrics of Decision trees on their own data.')
    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the uploaded CSV file
        combined_data = pd.read_csv(uploaded_file)
        combined_data = combined_data.fillna('')
    else:
        # Use the default dataset
        st.write("No file uploaded. Using default dataset from Pfizer(ATLAS) and Venatorx(GEARS) surveillance for 2022.")

    def analyze_resistance(data, filter_column='Source'):
        results = {}
        filter_values = data[filter_column].unique()

        def calculate_additional_metrics(youden_index, psi, accuracy):
            nnd = 1 / youden_index if youden_index != 0 else np.inf
            nnp = 1 / psi if psi != 0 else np.inf
            nnm = 1 / (1 - accuracy) if accuracy != 1 else np.inf
            return nnd, nnp, nnm

        def calculate_metrics(conf_matrix):
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
            youden_index = sensitivity + specificity - 1
            psi = ppv + npv - 1
            return sensitivity, specificity, ppv, npv, youden_index, psi

        for value in filter_values:
            filtered_data = data[data[filter_column] == value]
            X = filtered_data.drop(columns=['Resistance', filter_column])
            y = filtered_data['Resistance']
            X = X.apply(LabelEncoder().fit_transform)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            conf_matrix = confusion_matrix(y_test, y_pred)
            sensitivity, specificity, ppv, npv, youden_index, psi = calculate_metrics(conf_matrix)
            accuracy = accuracy_score(y_test, y_pred)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan

            nnd, nnp, nnm = calculate_additional_metrics(youden_index, psi, accuracy)
            results[value] = {
                'Confusion Matrix': conf_matrix,
                'Accuracy': accuracy,
                'AUC': auc,
                'Sensitivity (Recall)': sensitivity,
                'Specificity': specificity,
                'PPV (Precision)': ppv,
                'NPV': npv,
                'Youden Index (Y)': youden_index,
                'Predictive Summary Index (PSI or Ψ)': psi,
                'NND (Number Needed to Diagnose)': nnd,
                'NNP (Number Needed to Predict)': nnp,
                'NNM (Number Needed to Misdiagnose)': nnm
            }

        return results, filter_column

    if combined_data is not None:
        st.write("Performing resistance analysis...")
        results, filter_column = analyze_resistance(combined_data, 'Source')
        st.write(f"Analysis results based on {filter_column}:")

        for source, metrics in results.items():
            output = (
                f"For the source **{source}**, the analysis yielded an accuracy of **{metrics['Accuracy']:.2f}** "
                f"and an AUC of **{metrics['AUC']:.2f}**. The sensitivity (recall) was **{metrics['Sensitivity (Recall)']:.2f}**, "
                f"while the specificity reached **{metrics['Specificity']:.2f}**. The positive predictive value (PPV or precision) "
                f"was **{metrics['PPV (Precision)']:.2f}**, and the negative predictive value (NPV) was **{metrics['NPV']:.2f}**. "
                f"The Youden Index (Y) was calculated at **{metrics['Youden Index (Y)']:.2f}**, and the Predictive Summary Index (PSI or Ψ) "
                f"stood at **{metrics['Predictive Summary Index (PSI or Ψ)']:.2f}**. Additionally, the Number Needed to Diagnose (NND) "
                f"was **{metrics['NND (Number Needed to Diagnose)']:.2f}**, the Number Needed to Predict (NNP) was **{metrics['NNP (Number Needed to Predict)']:.2f}**, "
                f"and the Number Needed to Misdiagnose (NNM) was **{metrics['NNM (Number Needed to Misdiagnose)']:.2f}**."
            )
            st.write(output)

            # Radar chart for the metrics
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=[metrics['Sensitivity (Recall)'], metrics['Specificity'], metrics['PPV (Precision)'], metrics['NPV']],
                theta=['Sensitivity', 'Specificity', 'PPV', 'NPV'],
                fill='toself',
                name=source
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False
            )

            st.plotly_chart(fig)
