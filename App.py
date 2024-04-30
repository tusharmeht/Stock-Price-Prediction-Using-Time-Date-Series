# importing libraries

import pickle
from pathlib import Path
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

import streamlit as st 
import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go 
import plotly.express as px
import datetime 
from datetime import date, timedelta 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

# #login credentials
# def creds_entered():
#     if st.session_state["user"].strip() == "admin" and st.session_state['passwd'].strip()=="admin":
#         st.session_state['authenticated']  = True
#     else:
#         st.session_state['authenticated'] = False 
#         if not st.session_state['passwd']:
#             st.warning("Please Enter Password")
#         elif not st.session_state['user']:
#             st.warning("Please Enter Username")
#         else:
#             st.error("invalied Username/Password")

# def authenticate_user():
#     if "authenticated" not in st.session_state:
#         st.text_input(label='Username : ', value = "" , key= "user", on_change=creds_entered)
#         st.text_input(label='Password : ', value = "" , key= "passwd", type = 'password',on_change=creds_entered)
#         return False
#     else:
#         if st.session_state['authenticated']:
#             return True
#         else: 
#             st.text_input(label="Username :", value="", key="user", on_change=creds_entered)
#             st.text_input(label="Password :", value="", key="passwd",type = "password", on_change=creds_entered)
#             return False 

# if authenticate_user():

# ---USER AUTHENTICATION ---    
# names = ["Tushar Mehta", "Tanishq Bajaj", "Rajdeep Das"]
# usernames = ["tusharmehta3200","tanishqbajaj2002","rajdeepdas10"]
# passwords = ["mehta123","bajaj12","das22"]
# #load hashed passwords

# file_path = Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open("rb") as file:
#     hashed_passwords = pickle.load(file)

# authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "sales_dashboard","abcdef",cookie_expiry_days= 30)

# name, authentication_status, username = authenticator.login("Login", "main")
# if authentication_status == False:
#     st.error("Username/Password is incorrect")
# if authentication_status == None:
#     st.warning("Please enter your username and password ")
with open('D:\VS Code\config.yaml') as file:
    config = yaml.load(file, Loader = SafeLoader)
    
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login("main" )

if authentication_status:
    st.write(f"Welcome *{name}*")
elif authentication_status == False:
    st.error("Username/Password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
    
if authentication_status:
    #Title
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")

    app_name = 'Stock Market Forecasting App'
    st.title(app_name)
    st.subheader('This app is created to forecast the stock market price of this company')

    #add an image of stock market from an online resource
    st.image("image.jpg",width = 700)

    #take input from the user of the app about the start and the end date

    #making sidebar

    st.sidebar.header("Select the parameters from below")
    start_date = st.sidebar.date_input("Start Date", date(2020,1,1))
    end_date = st.sidebar.date_input("End Date", date(2024,2,10))

    #add ticker symbol list
    ticker_list = ["MSFT" , "AAPL" , "AMZN", "GOOG", "NVDA", "META" , "TSLA", "JPM", "WMT", "XOM", "JNJ", "PG", "AMD", "KO", "NFLX", "MCD", "CSCO", "BABA", "INTC", "IBM", "BA", "F", "C"]
    ticker = st.sidebar.selectbox("Select the Company",ticker_list)

    #fetch data from user inputs using yfinancen library

    data = yf.download(ticker, start= start_date, end = end_date)

    #add Date as a column to the dataframe

    data.insert(0, "Date", data.index, True)
    data.reset_index(drop=True, inplace = True)
    st.write('Data from ', start_date, 'to', end_date)
    st.write(data)


    #plot the data
    st.header("Data Visualization")
    st.subheader("Plot of the data")
    st.write("**NOTE:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific date")

    fig = px.line(data, x='Date', y = data.columns, title = "Closing Data", width= 900, height = 600, template = 'plotly_dark')
    st.plotly_chart(fig)

    #add a select box to select column data
    column = st.selectbox('Select the Column to be used to Forecasting ', data.columns[1:])

    #subsetting the data
    data = data[['Date', column]]
    st.write("Selected Data")
    st.write(data)

    #ADF test check stationarity
    st.header("Is Data Stationary? ")
    st.write(adfuller(data[column])[1]<0.05)

    #lets decompose the data
    st.header("Decompostion of the Data")
    decomposition = seasonal_decompose(data[column], model = 'additive', period = 12)
    st.write(decomposition.plot())

    #make same plot in plotly
    st.write(" PLotting the decomposition in plotly")
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title = 'Trend', width=800, height = 600, labels = {'x': 'Date', 'y': 'Price',}).update_traces(line_color='Green'))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title = 'Seasonal', width=800, height = 400, labels = {'x': 'Date', 'y': 'Price',}).update_traces(line_color='Blue'))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title = 'Residuals', width=800, height = 400, labels = {'x': 'Date', 'y': 'Price',}).update_traces(line_color='Red', line_dash='dot'))

    #Lets run the model
    #user input for three parameters of the model and seasonal order
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)

    seasonal_order = st.number_input("Select the value of seasonal p",0,24,12)
    #training model 
    model = sm.tsa.statespace.SARIMAX(data[column], order = (p,d,q), seasonal_order= (p,d,q,seasonal_order))
    model = model.fit(disp=-1)
    
    show_model_summary = False
    hide_model_summary = False
    if st.button("Show Model Summary"):
        #print model summary
        st.header('Model Summary')
        st.write(model.summary())
        st.write("---")
        show_model_summary=True
    else:
        show_model_summary=False
    
    if st.button("Hide Model Summary"):
        if not hide_model_summary:
            hide_model_summary=True
        else:
            hide_model_summary=False
        


    # predict the future values (Forecasting)
    st.write("<p style= 'color:green; font-size: 50: font-weight: bold;'>Forecasting the data </p>", unsafe_allow_html= True)
    forecast_period = st.number_input("Select the number of days to forecast ",1,365,10)
    #predict the future values 
    predictions = model.get_prediction(start= len(data), end= len(data)+forecast_period-1)
    predictions = predictions.predicted_mean
    # st.write(predictions)

    #add index to results dataframe as dates
    predictions.index = pd.date_range(start=end_date, periods = len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)


    predictions.insert(0, 'Date', predictions.index, True)
    predictions.reset_index(drop=True, inplace = True)

    st.write("## Predictions", predictions)
    st.write("## Actual Data", data)
    st.write("---")

    #lets plot the figure
    fig = go.Figure()
    #add actual data to the plot
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    #add predicted data to the plot
    fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
    #set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=800, height = 400)
    #display the plot
    st.plotly_chart(fig)



    #Add buttons to show and hide separate plots
    show_plots = False
    if st.button('Show Separate Plots'):
        if not show_plots:
            st.write(px.line(x=data['Date'], y=data[column],title= 'Actual', width=800, height = 400, labels = {'x': 'Date', 'y': 'Price'}).update_traces(line_color= 'Blue'))
            st.write(px.line(x=predictions['Date'],y=predictions['predicted_mean'], title = 'Predicted', width = 800, height = 400 ,labels = {'x': 'Date', 'y': 'Price'}).update_traces(line_color = 'Red'))
            
            show_plots = True
        else:
            show_plots = False 
        
    hide_plots = False
    if st.button("Hide Separate Plots"):
        if not hide_plots:
            hide_plots= True
            
        else :
            hide_plots = False
    
    st.write("---")