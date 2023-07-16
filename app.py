import streamlit as st
import pickle
import pandas as pd

# with open('styles.css') as f:
#     st.markdown(f'<style>{f.read()}</style>' , unsafe_allow_html=True)

# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://wallpapercave.com/wp/wp4059913.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
#
# add_bg_from_url()

teams = ['Mumbai Indians',
 'PKings XI Punjab',
 'Chennai Super Kings',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Sunrisers Hyderabad']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi', 'Chennai',
       'Jaipur', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Bengaluru']


pipe = pickle.load(open('pipe_logreg.pkl' , 'rb'))
st.title("IPL Win Predictor - by Ashitosh and Aaryan")
# with st.form(key='my_form'):

col1 , col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting team" , sorted(teams))
with col2:
    bowling_team = st.selectbox("Select Bowling team" , sorted(teams))

select_city = st.selectbox("Select Host City" , sorted(cities))

target = st.number_input("Target")

col3 , col4 , col5 = st.columns(3)

with col3:
    score = st.number_input("Score")
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input("Wickets out")

if st.button("Predict Probability"):
    runs_left = target  - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team' : [batting_team],
                             'bowling_team' : [bowling_team],
                             'city' : [select_city],
                             'runs_left' : [runs_left],
                             'balls_left' : [balls_left],
                             'wickets' : [wickets],
                             'total_runs_x' : [target] ,
                             'crr' : [crr],
                             'rrr' : [rrr]})


    result  = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss* 100)) + "%")







