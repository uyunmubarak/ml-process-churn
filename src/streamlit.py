import streamlit as st
import requests
import joblib
from PIL import Image

st.title("Churn Prediction")
# Load and set images in the first place
header_images = Image.open('../assets/churn.png')
st.image(header_images)

# Add some information about the service
st.subheader("Please fill variabel below then click Predict button.")

# Create form of input
with st.form(key = "churn_data_form"):
    # Create box for number input
    age = st.number_input(
        label = "1.Enter Age Value:",
        min_value = 10,
        max_value = 64,
        help = "Value range from 10 to 64"
    )

    days_since_last_login = st.number_input(
        label = "2.Enter Days Since Last Login Value:",
        min_value = -99,
        max_value = 26,
        help = "Value range from -999 to 26"
    )
    
    points_in_wallet = st.number_input(
        label = "3.Enter Average Time Spent Value:",
        min_value = -760.661236,
        max_value = 2069.069761,
        help = "Value range from -760.661236 to 2069.069761"
    )

    gender = st.selectbox(
        label = "4.Please Choose Gender:",
        options = (
            "F",
            "M",
            "Unknown"
        )
    )

    region_category = st.selectbox(
        label = "5.Please Choose Region Category:",
        options = (
            "City",
            "Town",
            "Village",
            ""
        )
    )

    membership_category = st.selectbox(
        label = "6.Please Choose Membership Category:",
        options = (
            "Basic Membership",
            "Silver Membership",
            "Platinum Membership",
            "Gold Membership",
            "No Membership",
            "Premium Membership"
        )
    )

    joined_through_referral = st.selectbox(
        label = "7.Please Choose Joined Through Referral:",
        options = (
            "Yes",
            "No",
            "?"
        )
    )

    preferred_offer_types = st.selectbox(
        label = "8.Please Choose Preffered Offer Types:",
        options = (
            "Gift Vouchers/Coupons",
            "Credit/Debit Card Offers",
            "Without Offers",
            ""
        )
    )

    medium_of_operation = st.selectbox(
        label = "9.Please Choose Medium of Operation:",
        options = (
            "Desktop",
            "Smartphone",
            "Both",
            "?"
        )
    )

    internet_option = st.selectbox(
        label = "10.Please Choose Internet Option:",
        options = (
            "Wi-Fi",
            "Mobile_Data",
            "Fiber_Optic"
        )
    )

    used_special_discount = st.selectbox(
        label = "11.Please Choose Used Special Discount:",
        options = (
            "Yes",
            "No",
        )
    )

    offer_application_preference = st.selectbox(
        label = "12.Please Choose Offer Application Preference:",
        options = (
            "Yes",
            "No",
        )
    )

    past_complaint = st.selectbox(
        label = "13.Please Choose Past Complaint:",
        options = (
            "Yes",
            "No",
        )
    )

    complaint_status = st.selectbox(
        label = "14.Please Choose Complaint Status:",
        options = (
            "Not Applicable",
            "Unsolved",
            "Solved",
            "Solved in Follow-up",
            "No Information Available"
        )
    )

    feedback = st.selectbox(
        label = "15.Please Choose Feedback:",
        options = (
            "Poor Product Quality",
            "No reason specified",
            "Too many ads",
            "Poor Website",
            "Poor Customer Service",
            "Reasonable Price",
            "User Friendly Website",
            "Products always in Stock",
            "Quality Customer Care"
        )
    )
    
    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "age": age,
            "days_since_last_login": days_since_last_login,
            "points_in_wallet": points_in_wallet,
            "gender": gender,
            "region_category": region_category,
            "membership_category": membership_category,
            "joined_through_referral": joined_through_referral,
            "preferred_offer_types": preferred_offer_types,
            "medium_of_operation": medium_of_operation,
            "internet_option": internet_option,
            "used_special_discount": used_special_discount,
            "offer_application_preference": offer_application_preference,
            "past_complaint": past_complaint,
            "complaint_status": complaint_status,
            "feedback":feedback
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict", json = raw_data).json()
            
        # Parse the prediction result
        if res["prediction"] == "0":
            st.success("NO CHURN!!!")

        else:
            st.warning("CHURN!!!")