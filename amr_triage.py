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
    page_icon="💊",  # Pill icon
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar expanded by default
)

selected_tab = option_menu(
    menu_title=None,
    options=["About AST Triage Webapp", "AST Triage Tool", "Performance of Decision Trees"],  # Added "About AST Triage Webapp"
    icons=["ℹ️", "🦠", "📊"],  # Added the information icon for the new tab
    orientation="horizontal",
    styles={
        "container": {"padding": "5px", "background-color": "#f0f0f0"},
        "icon": {"color": "#ADD8E6", "font-size": "24px"},  # Increase icon size
        "nav-link": {
            "font-size": "24px",  # Increase font size
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#189AB4",
        },
        "nav-link-selected": {"background-color": "#020659", "color": "white"},
    },
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


# Inject custom CSS to increase the font size of tabs
st.markdown("""
    <style>
    div[data-baseweb="tab"] > button {
        font-size: 40px !important;
    }
    </style>
    """, unsafe_allow_html=True)






if selected_tab == "About AST Triage Webapp":
    st.title("About AST Triage Webapp")
    st.markdown("""
    ### Welcome to the AST Triage Webapp!

    This web application is designed to assist healthcare professionals in prioritizing antimicrobial susceptibility testing (AST) and analyzing the performance of decision trees on resistance data. Below you will find detailed information about the webapp, including its development, datasets used, and the theoretical foundation of decision trees.

    #### Key Features:
    - **AST Triage Tool**: Helps in assessing the urgency of AST based on patient details.
    - **Performance of Decision Trees**: Provides tools to policy makers for evaluation of decision trees in AST triaging.

    #### Developed By:
    - Dr. Oscar Nyangiri
    - Dr. Primrose Beryl
    - Dr. Fred Mutisya

    #### Data Sources:
    - Pfizer Atlas Data
    - Venatorx Gears Data

    This tool was developed as part of the Vivli AMR Data Challenge 2024.

    **Disclaimer**: The tool is intended for informational purposes only and should not replace professional medical judgment.
    """)

    st.markdown("### Learn More About Each Section:")

    with st.expander("1. AMR Data Challenge and Vivli"):
        st.markdown("""
        The Vivli AMR Data Challenge is an initiative that aims to tackle the global threat of antimicrobial resistance (AMR) by leveraging data sharing and collaboration. Vivli is an organization dedicated to enabling open data sharing and maximizing the reuse of clinical trial data. The AMR Data Challenge focuses on encouraging the use of shared data to develop innovative tools and solutions to combat AMR.
        
        This challenge invites researchers and data scientists to analyze data related to AMR and develop tools that can support healthcare decision-making, policy development, and research. The AST Triage Webapp was developed as part of this challenge, utilizing datasets provided by Pfizer and Venatorx.
        """)

    with st.expander("2. About the AMR Datasets - Pfizer (ATLAS) and Venatorx (GEARS)"):
        st.markdown("""
        The AMR datasets used in this webapp include:
        
        **Pfizer ATLAS**: The Antimicrobial Testing Leadership and Surveillance (ATLAS) database is a comprehensive global resource that tracks antimicrobial resistance trends. It includes data on various pathogens and their resistance patterns across different regions.
        
        **Venatorx GEARS**: The Global Essential Antimicrobial Resistance Surveillance (GEARS) dataset from Venatorx provides detailed information on resistance mechanisms, focusing on critical pathogens and the efficacy of antibiotics.
        
        These datasets are crucial for developing models and tools that help predict resistance patterns and support informed decision-making in clinical settings.
        """)

    with st.expander("3. The Antibiogram Structure and Filtering"):
        st.markdown("""
        The webapp generates customized antibiograms based on the patient's details and the selected criteria, such as country and sample source. The filtering mechanism includes:
        
        - **Country**: The data is first filtered by the selected country.
        - **Bordering Countries**: If no data is found for the selected country, the app checks for neighboring countries that share borders(hence epidemiological linkages) and uses their data.
        - **Region**: If data is still not available, the app broadens the search to include countries in the same WHO region.
        
        The antibiogram provides a summary of the susceptibility of various species to different antibiotics, helping clinicians make informed decisions about treatment.
        """)

    with st.expander("4. Decision Tree Analysis and Theory"):
        st.markdown("""
        Decision trees are a popular machine learning method used for classification and regression tasks. In the context of antimicrobial susceptibility testing (AST), decision trees can help predict the likelihood of resistance based on patient data and historical patterns.
        
        **How Decision Trees Work**:
        - **Nodes**: Represent the features in the dataset.
        - **Branches**: Represent the decision rules.
        - **Leaves**: Represent the outcomes (e.g., susceptible or resistant).
        
        The model learns by splitting the data at each node based on the feature that provides the highest information gain (or lowest Gini impurity). This process continues until the tree reaches its maximum depth or can no longer improve the classification.
        
        **Why Use Decision Trees for AST?**:
        - **Interpretability**: Easy to understand and interpret the model's decisions.
        - **Flexibility**: Can handle both numerical and categorical data.
        - **No Need for Feature Scaling**: Decision trees do not require normalization or scaling of data.
        """)
    

    with st.expander("5. Performance of the Model for Policy Makers"):
        st.markdown("""
        Understanding the performance of decision trees in the context of AST is critical for policy makers. Model performance can vary with different body sample sources. Here are some key metrics used to evaluate the model:
        """)
    
        st.write("**Accuracy**: The ratio of correctly predicted instances to the total instances.")
        st.latex(r'''
        \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
        ''')
    
        st.write("**Sensitivity (Recall)**: The ability of the model to correctly identify true positives.")
        st.latex(r'''
        \text{Sensitivity} = \frac{TP}{TP + FN}
        ''')
    
        st.write("**Specificity**: The ability of the model to correctly identify true negatives.")
        st.latex(r'''
        \text{Specificity} = \frac{TN}{TN + FP}
        ''')
    
        st.write("**Precision (PPV)**: The proportion of true positives among the instances classified as positive.")
        st.latex(r'''
        \text{Precision} = \frac{TP}{TP + FP}
        ''')
    
        st.write("**Negative Predictive Value (NPV)**: The proportion of true negatives among the instances classified as negative.")
        st.latex(r'''
        \text{NPV} = \frac{TN}{TN + FN}
        ''')
    
        st.write("**Youden Index**: A summary measure of the effectiveness of a diagnostic test.")
        st.latex(r'''
        \text{Youden Index} = \text{Sensitivity} + \text{Specificity} - 1
        ''')
    
        st.write("**Predictive Summary Index (PSI)**: Combines precision and NPV to evaluate predictive performance.")
        st.latex(r'''
        \text{PSI} = \text{Precision} + \text{NPV} - 1
        ''')
    
        st.write("**Number Needed to Diagnose (NND)**: The number of patients that need to be tested to correctly diagnose one patient.")
        st.latex(r'''
        \text{NND} = \frac{1}{\text{Youden Index}}
        ''')
    
        st.write("**Number Needed to Predict (NNP)**: The number of predictions needed to correctly predict one positive case.")
        st.latex(r'''
        \text{NNP} = \frac{1}{\text{PSI}}
        ''')
    
        st.write("**Number Needed to Misdiagnose (NNM)**: The number of predictions needed to produce one incorrect prediction.")
        st.latex(r'''
        \text{NNM} = \frac{1}{1 - \text{Accuracy}}
        ''')
    
        st.markdown("""
        These metrics help assess the reliability and robustness of the decision tree model in predicting antimicrobial resistance, supporting informed decision-making for health policies.
        """)









elif selected_tab == "AST Triage Tool":
    # AST Prioritization Tool Interface
    st.title('Antimicrobial Susceptibility Testing (AST) Triage Tool')
    st.markdown("""
    <div style="background-color:#484a4d;padding:10px">
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






    
    
    
    # Function to format colors based on susceptibility percentage
    def color_format(val):
        if isinstance(val, str) and val.endswith('%'):
            percentage = float(val.rstrip('%'))
            if percentage > 75:
                return 'background-color: green'
            elif 50 <= percentage <= 75:
                return 'background-color: orange'
            else:
                return 'background-color: red'
        return ''
    
    # Function to calculate and display antibiogram
    def display_antibiogram(filtered_data, message):
        if not filtered_data.empty:
            st.write(message)
            filtered_data['Species'] = filtered_data['Species'].astype(str).fillna('')
            filtered_data['Antibiotics'] = filtered_data['Antibiotics'].astype(str).fillna('')
            filtered_data['Resistance'] = pd.to_numeric(filtered_data['Resistance'], errors='coerce').fillna(0).astype(int)
    
            resistance_summary = filtered_data.groupby(['Species', 'Antibiotics', 'Resistance']).size().unstack(fill_value=0)
            if 0 in resistance_summary.columns:
                resistance_summary['Total Count'] = resistance_summary.sum(axis=1)
                resistance_summary['% Susceptibility'] = (resistance_summary[0] / resistance_summary['Total Count']) * 100
                resistance_summary['% Susceptibility'] = resistance_summary['% Susceptibility'].round(1).apply(lambda x: f"{x:.1f}%")
            else:
                st.write("No susceptible data available to calculate '% Susceptibility'.")
    
            pivoted_susceptibility = resistance_summary.pivot_table(index='Species', columns='Antibiotics', values='% Susceptibility', aggfunc='first')
            all_antibiotics = combined_data['Antibiotics'].unique()
            pivoted_susceptibility = pivoted_susceptibility.reindex(columns=all_antibiotics, fill_value='N/A')
            pivoted_susceptibility.insert(0, 'Total Count', resistance_summary.groupby('Species')['Total Count'].first())
    
            styled_summary = pivoted_susceptibility.style.applymap(color_format, subset=pd.IndexSlice[:, pivoted_susceptibility.columns[1:]])
            st.write("Detailed Antibiogram")
            st.dataframe(styled_summary)
        else:
            st.write(f"No data found. {message}")



    with col1:
        if st.button('Customised Triage Antibiogram'):
            if filtered_data.empty:
                bordering_countries = world_bank_data.loc[world_bank_data['Country'] == country, 'Bordering Countries'].values
                if bordering_countries.size > 0 and isinstance(bordering_countries[0], str):
                    bordering_countries_list = bordering_countries[0].split(', ')
                    filtered_data = combined_data.loc[(combined_data['Country'].isin(bordering_countries_list)) & (combined_data['Source'] == source)]
                    message = f"No data for {country}and {source} for triage antibiogram. Using bordering countries: {', '.join(bordering_countries_list)}"
                else:
                    filtered_data = combined_data.loc[(combined_data['Source'] == source)]
                    message = f"No data for {country} and {source} or its borders. Using the full dataset to make custom triage antibiogram."
            else:
                message = f"Triage antibiogram made using filtering criteria of: {country} and {source}"
    
            display_antibiogram(filtered_data, message)

    



    
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


















# Create the second tab
elif selected_tab == "Performance of Decision Trees":
    st.title("Performance of the Decision Tree Model")
    st.write('This is a tool for policy makers to test the performance metrics of Decision Trees on their own data.')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the uploaded CSV file
        combined_data = pd.read_csv(uploaded_file)
        combined_data = combined_data.fillna('')
    else:
        # Use the default dataset
        st.write("No file uploaded. Using default dataset from Pfizer(ATLAS) and Venatorx(GEARS) surveillance for 2022.")
        # Assume the combined_data is loaded here (replace with your actual dataset if necessary)
    
    if combined_data is not None:
        # Allow the user to select the source from a dropdown menu
        selected_source = st.selectbox("Select the source:", combined_data['Source'].unique())
        
        # Filter data based on the selected source
        filtered_data = combined_data[combined_data['Source'] == selected_source]

        def analyze_resistance(data):
            results = {}

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

            X = data.drop(columns=['Resistance', 'Source'])
            y = data['Resistance']
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
            results[selected_source] = {
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

            return results

        if not filtered_data.empty:
            st.write(f"Performing resistance analysis for source: {selected_source}")
            results = analyze_resistance(filtered_data)
            metrics = results[selected_source]

            output = (
                f"For the source **{selected_source}**, the analysis yielded an accuracy of **{metrics['Accuracy']:.2f}** "
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
                name=selected_source
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
        else:
            st.write(f"No data available for the selected source: {selected_source}")








