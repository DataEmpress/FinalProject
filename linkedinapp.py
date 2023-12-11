
#Christina Brown
#OPAN 6607 Programming II: Final Project (LinkedIn Machine Learning Streamlit App)
#Fall 2023, Professor Lyon
#December 10, 2023

#Python Libraries Uploaded
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#The ss dataset and the logistics regression model from the Juputer notebook. 

s = pd.read_csv("socialmediausage.csv")
ss = pd.DataFrame({
          "sm_li":np.where(s["web1h"] == 1, 1, 0),
          "income":np.where(s["income"] > 9, np.nan, s["income"]),        
          "education":np.where(s["educ2"]> 8, np.nan, s["educ2"]),
          "parent":np.where(s["par"] == 1, 1, 0),
          "married":np.where(s["marital"] == 1, 1, 0),
          "female":np.where(s["gender"] == 1, 1, 0), 
          "age": np.where(s["age"] > 98, np.nan, s["age"])})
 
#Dropped any missing values from the ss DataFrame.
ss = ss.dropna()
 
#Split the data into testing and training datasets.     
y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]    
x_train, x_test, y_train, y_test = train_test_split (x,
                                                     y,
                                                     stratify = y, 
                                                     test_size = 0.2, 
                                                     random_state = 987)    
 #Initiate the logistic regression algorithm.
log_Res = LogisticRegression()
class_weight = "balanced"

 #Fit the logistic regression algorithm to x_train and y_train.
log_Res.fit(x_train, y_train) 
    
#Make predictions using the model.
y_pred = log_Res.predict(x_test)

#The logisitic regression model is attached to the Predict button in the Streamlit app.
#That way, the moment the user hits that button, they will get their results.

st.title("Accenture Labs' LinkedIn Usage Simulation")
st.write("At Accenture, our analytics team is closely analyzing social media behavior in the U.S.")
st.write("LinkedIn has become the **primier social media platform** for job seekers in an oversalutated job market.")
st.write("With your responses below, we can **accurately predict** if you're a LinkedIn user or not. ")
st.write("Don't believe us? Let us show you how!")

st.divider()

#Question 1: Have the user enter their name.
username = st.text_input("What's your first name?")

#Question #2: Asking user for income level information.
income = st.selectbox ("Annual Income Level", 
                          options = ["Less than $10,000 a year", 
                                     "$10 to under $20,000 a year",
                                     "$20 to under $30,000 a year",
                                     "$30 to under $40,000 a year",
                                     "$40 to under $50,000 a year",
                                     "$50 to under 75,000 a year",
                                     "$75 to under $100,000 a year",
                                     "$100 to under $150,000 a year",
                                     "More than $150,000 a year"])

if income == "Less than $10,000 a year":
    income = 1
elif income == "$10 to under $20,000 a year":
    income = 2
elif  income == "$20 to under $30,000 a year":
     income  = 3
elif  income  == "$30 to under $40,000 a year":
     income = 4
elif  income  == "$40 to under $50,000 a year":
     income  = 5
elif  income  == "$50 to under 75,000 a year":
     income  = 6
elif  income  == "$75 to under $100,000 a year":
    income  = 7
elif income  == "$100 to under $150,000 a year":
     income  = 8
elif income == "More than $150,000 a year":
     income = 9 

#Question #2: Asking for their education level. 
education = st.selectbox("Education Level", 
                          options = ["Less than a high School diploma (Grades 1-8)",
                                     "High School diploma incomplete (Grades 9-12 without a diploma)",
                                     "High School Diploma",
                                     "Some college education, no degree",
                                     "Two-year Associate Degree",
                                     "Four-year College Degree",
                                     "Some postgraduate or professional schooling",
                                     "Postgraduate or professional degree"])

if education == "Less than a high School diploma (Grades 1-8)":
    education = 1
elif education == "High School diploma incomplete (Grades 9-12 without a diploma)":
    education = 2
elif education == "High School Diploma":
    education = 3
elif education == "Some college education, no degree":
    education = 4
elif education == "Two-year Associate Degree":
    education = 5
elif education == "Four-year College Degree":
    education = 6
elif education == "Some postgraduate or professional schooling":
    education = 7
elif education == "Postgraduate or professional degree":
    education = 8  

#Question #3: Asking the user whether or not they are a parent.
parent = st.selectbox ("Are you a parent or a guardian?", 
                          options = ["Yes",
                                     "No"])
if parent == "Yes":
    parent = 1
elif parent == "No":
    parent = 0 

#Question #4: Asking for the user's current marital status.
married = st.selectbox ("Martial Status", 
                          options = ["Married", "Not Married"])
if married == "Married":
    married = 1
elif married == "Not Married":
    married = 0
    
#Question #5: Asking user for their gender identity.
female = st.radio ("Gender", 
                          options = ["Female", "Male"])

if female == "Female":
    female = 1
elif female == "Male": 
    female = 0 

#Question 6: Asking user for their age.
age = st.slider (label = "Age",
                 min_value = 1,
                 max_value = 100,
                 value = 50)

#The Submit Button 
ss= st.button('Predict') 
if ss: 
    new_predictions = [income, education, parent, married, female, age]
    predicted_class = log_Res.predict([new_predictions])
    probability = log_Res.predict_proba([new_predictions])
    st.success(f"Predicted class:{predicted_class[0]}")
    st.success(f"{username}, you have a {probability[0][1]} chance of being a LinkedIn user!")



st.write("@2023 Powered By: Christina Brown from Cohort 4 of Georgetown's MSBA program. Hoya!")

