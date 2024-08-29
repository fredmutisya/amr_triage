import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

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

# AST Prioritization Tool Interface
st.title('Antimicrobial Susceptibility Testing (AST) Prioritization Tool')
st.markdown("""
<div style="background-color:#ADD8E6;padding:10px">
<h2 style="color:white;text-align:center;">Please answer the following questions about the patient:</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.info("""
Input the patient details into the AST triage tool to assess the urgency of antimicrobial resistance testing.
""")

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
    #st.write("No exact match found. Relaxing criteria...")
    
    # Relax criteria: Remove Source filter
    filtered_data = combined_data[
        (combined_data['Country'] == country) &
        (combined_data['Gender'] == gender) &
        (combined_data['Speciality'] == speciality) &
        (combined_data['Antibiotics'] == antibiotic) &
        (combined_data['Age.Group'] == age)
    ]
    final_criteria = {'Country': country, 'Gender': gender, 'Speciality': speciality, 'Antibiotics': antibiotic, 'Age.Group': age}
    
    if filtered_data.empty:
        #st.write("Relaxing criteria further...")

        # Remove Speciality filter
        filtered_data = combined_data[
            (combined_data['Country'] == country) &
            (combined_data['Gender'] == gender) &
            (combined_data['Antibiotics'] == antibiotic) &
            (combined_data['Age.Group'] == age)
        ]
        final_criteria = {'Country': country, 'Gender': gender, 'Antibiotics': antibiotic, 'Age.Group': age}
        
        if filtered_data.empty:
            #st.write("Relaxing criteria further...")
            
            # Remove Age.Group filter
            filtered_data = combined_data[
                (combined_data['Country'] == country) &
                (combined_data['Gender'] == gender) &
                (combined_data['Antibiotics'] == antibiotic)
            ]
            final_criteria = {'Country': country, 'Gender': gender, 'Antibiotics': antibiotic}

            if filtered_data.empty:
                #st.write("Relaxing criteria further...")

                # Remove Gender filter
                filtered_data = combined_data[
                    (combined_data['Country'] == country) &
                    (combined_data['Antibiotics'] == antibiotic)
                ]
                final_criteria = {'Country': country, 'Antibiotics': antibiotic}

                if filtered_data.empty:
                    #st.write("Relaxing criteria further...")

                    # Use entire dataset
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
if st.button('Show AST Model Result'):
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

