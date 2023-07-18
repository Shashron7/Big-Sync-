import streamlit as st
from streamlit_option_menu import option_menu
import sqlalchemy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
#Database Utility Class
from sqlalchemy.engine import create_engine
# Provides executable SQL expression construct
from sqlalchemy.sql import text
import toml
from plotly.subplots import make_subplots


# Reading data
toml_data = toml.load(r"C:\Users\sshas\Desktop\Dashboard\.streamlit\secrets.toml")
# toml_data = toml.load("C:/Users/HP/.streamlit/secrets.toml")
# saving each credential into a variable
HOST_NAME = toml_data['postgres']['host']
DATABASE = toml_data['postgres']['dbname']
PASSWORD = toml_data['postgres']['password']
USER = toml_data['postgres']['user']
PORT = toml_data['postgres']['port']


# Using the variables we read from secrets.toml
# mydb = connection.connect(host=HOST_NAME, database=DATABASE, user=USER, passwd=PASSWORD, use_pure=True)

class PostgresqlDB:
    def __init__(self,user_name,password,host,port,db_name):
        self.user_name = user_name
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.engine = self.create_db_engine()

    def create_db_engine(self):
        try:
            db_uri = 'postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{db_name}'.format(
                      user_name=self.user_name,password=self.password,
                      host=self.host,db_name=self.db_name,port=self.port)
            return create_engine(db_uri)
        except Exception as err:
            raise RuntimeError(f'Failed to establish connection -- {err}') from err

    def execute_dql_commands(self,stmnt,values=None):
  
        try:
            with self.engine.connect() as conn:
                if values is not None:
                    result = conn.execute(stmnt,values)
                else:
                    result = conn.execute(stmnt)
            return result
        except Exception as err:
            print(f'Failed to execute dql commands -- {err}')
    
    def execute_ddl_and_dml_commands(self,stmnt,values=None):
        connection = self.engine.connect()
        trans = connection.begin()
        try:
            if values is not None:
                result = connection.execute(stmnt,values)
            else:
                result = connection.execute(stmnt)
            trans.commit()
            connection.close()
            print('Command executed successfully.')
        except Exception as err:
            trans.rollback()
            print(f'Failed to execute ddl and dml commands -- {err}')



#Defining Db Credentials
USER_NAME = 'postgres'
PASSWORD = 'footystar7'
PORT = 5432
DATABASE_NAME = 'newdb'
HOST = 'localhost'

#Note - Database should be created before executing below operation
#Initializing SqlAlchemy Postgresql Db Instance
db = PostgresqlDB(user_name=USER_NAME,
                    password=PASSWORD,
                    host=HOST,port=PORT,
                    db_name=DATABASE_NAME)




st.title("POWER SYSTEM DATA")
st.write("QUERIES RELATED TO POWER SYSTEM DATA ARE DONE IN THIS PAGE")

display=0


selected1= option_menu(
    menu_title ="Select the query you need from below",
    options=["SHOW THE INPUT WAVEFORM"],)




st.title("POWER SYSTEM DATA")
st.write("QUERIES RELATED TO POWER SYSTEM DATA ARE DONE IN THIS PAGE")

# Add an option to exit the app
exit_option = "Exit the app"

# Define the options for the option_menu
options = ["SHOW THE INPUT WAVEFORM", "SHOW THE OUTPUT WAVEFORM",exit_option]

# Get the selected option from the option_menu
st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select what graph you want to see** :</span>", unsafe_allow_html=True)
selected1 = st.selectbox("", options)

# Check if the exit option is selected
if selected1 == exit_option:
    st.warning("Exiting the app...")
    st.stop()

if selected1 == "SHOW THE INPUT WAVEFORM":
    select_query_stmnt1 = text("SELECT * FROM input LIMIT 240 ;") 

    result_1 = db.execute_dql_commands(select_query_stmnt1)

    result_1 = pd.DataFrame(result_1)
    
elif selected1=="SHOW THE OUTPUT WAVEFORM":
    select_query_stmnt1 = text("SELECT * FROM output LIMIT 240 ;") 

    result_1 = db.execute_dql_commands(select_query_stmnt1)

    result_1 = pd.DataFrame(result_1)
        
st.dataframe(result_1)


# opt=["Input Graph", "Output Graph"]
# st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select what statistical feature you want to calculate** :</span>", unsafe_allow_html=True)
# vis_selected=st.selectbox("", options=opt)

phasea=result_1["Phase_A_current"].values
phaseb=result_1["Phase_B_current"].values
phasec=result_1["Phase_C_current"].values
time=result_1["Timestamp"].values

fig = make_subplots(rows=3, cols=1,vertical_spacing=0.3)
fig.add_trace(go.Scatter(x=np.arange(len(phasea)), y=phasea, mode='markers', name='Phase A current'), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(len(phaseb)), y=phaseb, mode='markers', name='Phase B current'), row=2, col=1)
fig.add_trace(go.Scatter(x=np.arange(len(phasec)), y=phasec, mode='markers', name='Phase C current'), row=3, col=1)
fig.update_layout(
height=1200, width=800,
title="Visualisation of Phase current Graphs"
)
for i in range(1, 4):
        fig.update_xaxes(
        range=[0, 50],  # Set the initial range of the x-axis
        row=i, col=1,
        rangeslider=dict(visible=True)  # Enable the rangeslider for each subplot
    )
st.plotly_chart(fig)
fig.write_image("visual.png")

 
