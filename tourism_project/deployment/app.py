import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id= "CRR79/TourismPackage-Purchase-Prediction", filename="best_TourismPackage_Purchase_model_v1.joblib")
print("Model path:", model_path)

model = joblib.load(model_path)


# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of purchaging tourism package.
Please enter data below to get a prediction.
""")

# User input

age = st.number_input("Age", min_value=18, max_value=100, value=35)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=3)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children Visiting (<5 years)", min_value=0, max_value=5, value=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=50000)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 4)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Queen"])
num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)


# Derived features
family_size = num_persons + num_children
income_per_person = monthly_income / (family_size + 1)


# Prepare dataframe
input_data = pd.DataFrame([{
'Age': age,
'TypeofContact': type_of_contact,
'CityTier': city_tier,
'Occupation': occupation,
'Gender': gender,
'NumberOfPersonVisiting': num_persons,
'PreferredPropertyStar': property_star,
'MaritalStatus': marital_status,
'NumberOfTrips': num_trips,
'Passport': passport,
'OwnCar': own_car,
'NumberOfChildrenVisiting': num_children,
'Designation': designation,
'MonthlyIncome': monthly_income,
'PitchSatisfactionScore': pitch_score,
'ProductPitched': product_pitched,
'NumberOfFollowups': num_followups,
'DurationOfPitch': duration_pitch,
'FamilySize': family_size,
'IncomePerPerson': income_per_person
}])

# Print nicely
print("Input Data:")
print(input_data)

st.write("### Input DataFrame")

if st.button("Predict Purchage"):
    prediction = model.predict(input_data)
    print (" Prediction:",prediction[0])
    result = "Purchased" if prediction[0] == 1 else "Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
