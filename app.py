from MomentumLibrary import *
import streamlit as st
import datetime

st.title('Wakehurst Momentum')

train_start = st.date_input("Training start date",datetime.date(2015, 1, 1))
train_start = train_start.strftime('%d/%m/%Y')
# train_start2 = st.date_input(label="Data Start Date",value=datetime.date(2015,1,1))

today = datetime.datetime.now().strftime("%d/%m/%Y")
st.write(f"Today: {today}")

if st.button('Download Historical Data For All Assets'):
	scrape(list(assets.values()),train_start,today)