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

# Load saved model
model = pickle.load(open("decision_tree_model.pkl", 'rb'))

# Load the Variables CSV file for input options
Variables = pd.read_csv("Variables.csv")
Variables = Variables.fillna('')

# Load the main dataset from the provided CSV file
main_data_file = "combined_gears_atlas.csv"
combined_data = pd.read_csv(main_data_file)
combined_data = combined_data.fillna('')

# Extract predictor names from the Variables CSV file
countries = Variables["Country"].tolist()
source_sample = Variables["Source"].tolist()
antibiotics = Variables["Antibiotics"].tolist()
Speciality = Variables["Speciality"].tolist()

# Set up the tabs
tab1, tab2 = st.tabs(["AST Prioritization Tool", "Resistance Analysis"])

with tab1:
    # AST Prioritization Tool Interface
    st.title('Antimicrobial Susceptibility Testing (AST) Triage Tool')
    st.markdown("""
    <div style="background-color:#ADD8E6;padding:10px">
    <h2 style="color:white;text-align:center;">Please answer the following questions about the patient:</h2>
    </div>
    """, unsafe_allow_html=True)

    # New patient-related input fields as questions
    age = st.selectbox('What is the age of the patient?', ['0 to 2 Years','3 to 12 Years', '13 to 18 Years', '19 to 64 Years' ,'65 to 84 Years','85 and Over'])
    gender = st.selectbox('What is the gender of the patient?', ['Male', 'Female', 'Other'])
    speciality  = st.selectbox('What is the speciality of the patient?', Speciality)
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
    antibiotic = st.selectbox('Which antibiotics were administered before specimen collection?', antibiotics)

    # Filter the combined dataset based on the selected criteria
    filtered_data = combined_data[
        (combined_data['Country'] == country) &
        (combined_data['Gender'] == gender) &
        (combined_data['Source'] == source) &
        (combined_data['Speciality'] == speciality) &
        (combined_data['Antibiotics'] == antibiotic) &
        (combined_data['Age.Group'] == age)
    ]

    # Track the final criteria used
    final_criteria = {}

    # If no exact match, relax the criteria step by step
    if filtered_data.empty:
        filtered_data = combined_data[
            (combined_data['Country'] == country) &
            (combined_data['Gender'] == gender) &
            (combined_data['Speciality'] == speciality) &
            (combined_data['Antibiotics'] == antibiotic) &
            (combined_data['Age.Group'] == age)
        ]
        final_criteria = {'Country': country, 'Gender': gender, 'Speciality': speciality, 'Antibiotics': antibiotic, 'Age.Group': age}
    
        if filtered_data.empty:
            filtered_data = combined_data[
                (combined_data['Country'] == country) &
                (combined_data['Gender'] == gender) &
                (combined_data['Antibiotics'] == antibiotic) &
                (combined_data['Age.Group'] == age)
            ]
            final_criteria = {'Country': country, 'Gender': gender, 'Antibiotics': antibiotic, 'Age.Group': age}
        
            if filtered_data.empty:
                filtered_data = combined_data[
                    (combined_data['Country'] == country) &
                    (combined_data['Gender'] == gender) &
                    (combined_data['Antibiotics'] == antibiotic)
                ]
                final_criteria = {'Country': country, 'Gender': gender, 'Antibiotics': antibiotic}

                if filtered_data.empty:
                    filtered_data = combined_data[
                        (combined_data['Country'] == country) &
                        (combined_data['Antibiotics'] == antibiotic)
                    ]
                    final_criteria = {'Country': country, 'Antibiotics': antibiotic}

                    if filtered_data.empty:
                        filtered_data = combined_data.copy()
                        final_criteria = {}

    # Generate subgroup criteria based on the filtered data
    if not filtered_data.empty:
        for column in ['Country', 'Gender', 'Source', 'Speciality', 'Antibiotics', 'Age.Group']:
            if len(filtered_data[column].unique()) == 1:
                final_criteria[column] = filtered_data[column].iloc[0]

        top_species = filtered_data['Species'].value_counts().head(3).index.tolist()
        species_resistance = filtered_data.groupby('Species')['Resistance'].mean().loc[top_species]
    
        if final_criteria:
            criteria_str = ', '.join([f'{key}: {value}' for key, value in final_criteria.items()])
        else:
            criteria_str = "Entire dataset filtered by 'Country', 'Gender', 'Source', 'Speciality', 'Antibiotics', 'Age.Group'"

        st.write(f"Top 3 most common bacterial species and their resistance levels (Criteria: {criteria_str}):")
    
        for species, resistance in species_resistance.items():
            resistance_percentage = resistance * 100
            st.write(f"**{species}**: Resistance Level: **{resistance_percentage:.2f}%**")
    else:
        st.write("No data available even after relaxing the criteria.")

    # Prediction code
    susceptibility = ''
    if st.button('Show AST Triage Result'):
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
