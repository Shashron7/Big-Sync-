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
PASSWORD = ''
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
selected1 = st.selectbox("Select the action you want to perform", options)

# Check if the exit option is selected
if selected1 == exit_option:
    st.warning("Exiting the app...")
    st.stop()

if selected1 == "SHOW THE INPUT WAVEFORM":
    select_query_stmnt1 = text("SELECT * FROM input LIMIT 240 ;") 

    result_1 = db.execute_dql_commands(select_query_stmnt1)

    result_1 = pd.DataFrame(result_1)

elif selected1=="SHOW THE OUTPUT WAVEFORM":
    select_query_stmnt1 = text("SELECT * FROM output LIMIT 100 ;") 

    result_1 = db.execute_dql_commands(select_query_stmnt1)

    result_1 = pd.DataFrame(result_1)



phasea=result_1["Phase_A_current"].values
phaseb=result_1["Phase_B_current"].values
phasec=result_1["Phase_C_current"].values
time=result_1["Timestamp"].values

meana=np.mean(phasea)
meanb=np.mean(phaseb)
meanc=np.mean(phasec)

vara=np.var(phasea)
varb=np.var(phaseb)
varc=np.var(phasec)

snra = 10 * np.log10(np.mean(phasea ** 2) / np.mean((phasea - np.mean(phasea)) ** 2))
snrb = 10 * np.log10(np.mean(phaseb ** 2) / np.mean((phaseb - np.mean(phaseb)) ** 2))
snrc = 10 * np.log10(np.mean(phasec ** 2) / np.mean((phasec - np.mean(phasec)) ** 2))



st.dataframe(result_1)
stat_options=["Mean","Variance","Signal to Noise Ratio"]
st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select what statistical feature you want to calculate** :</span>", unsafe_allow_html=True)
stat_selected=st.selectbox("", options=stat_options)

if stat_selected=="Mean":
    st.text("The mean of the phase A current is " + str(meana))
    st.text("The mean of the phase B current is " + str(meanb))
    st.text("The mean of the phase C current is " + str(meanc))
    
    fig = make_subplots(rows=3, cols=1,vertical_spacing=0.3)
    fig.add_trace(go.Scatter(x=np.arange(len(phasea)), y=phasea, mode='markers', name='Phase A current'), row=1, col=1)
    fig.add_shape(
    type="line",
    x0=0,
    y0=meana,
    x1=len(phasea) - 1,
    y1=meana,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=1,
    col=1
    )
    
    fig.add_trace(go.Scatter(x=np.arange(len(phaseb)), y=phaseb, mode='markers', name='Phase B current'), row=2, col=1)
    fig.add_shape(
    type="line",
    x0=0,       
    y0=meanb,
    x1=len(phaseb) - 1,
    y1=meanb,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=2,
    col=1
    )
    
    fig.add_trace(go.Scatter(x=np.arange(len(phasec)), y=phasec, mode='markers', name='Phase C current'), row=3, col=1)
    fig.add_shape(
    type="line",
    x0=0,
    y0=meanc,
    x1=len(phasec) - 1,
    y1=meanc,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=3,
    col=1
    )
    fig.update_layout(
    height=1200, width=800,
    title="Mean of Phase current Graph"
    )
    fig.update_yaxes(title_text='Phase C current', row=3, col=1)
    fig.update_yaxes(title_text='Phase B current', row=2, col=1)
    fig.update_yaxes(title_text='Phase A current', row=1, col=1)
    for i in range(1, 4):
        fig.update_xaxes(
        range=[0, 100],  # Set the initial range of the x-axis
        row=i, col=1,
        rangeslider=dict(visible=True)  # Enable the rangeslider for each subplot
    )
    st.plotly_chart(fig)
    fig.write_image("figure.png")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 12))

    # ax1.plot(phasea, label='Data')
    # ax1.axhline(meana, color='r', linestyle='--', label='Mean')
    # ax1.legend()
    # ax1.set_title('Mean Plot of Phase A')

    # ax2.plot(phaseb, label='Data')
    # ax2.axhline(meanb, color='g', linestyle='--', label='Mean')
    # ax2.legend()
    # ax2.set_title('Mean Plot 2 of Phase B')

    # ax3.plot(phasec, label='Data')
    # ax3.axhline(meanc, color='b', linestyle='--', label='Mean')
    # ax3.legend()
    # ax3.set_title('Mean Plot 3 of Phase C')

    # Adjust spacing between subplots

    # Display the plots in Streamlit

    
elif stat_selected=="Variance":
    st.text("The Variance of the phase A current is " + str(vara))
    st.text("The Variance of the phase A current is " + str(varb))
    st.text("The Variance of the phase A current is " + str(varc))   
    
    
    fig = make_subplots(rows=3, cols=1,vertical_spacing=0.3)
    fig.add_trace(go.Scatter(x=np.arange(len(phasea)), y=phasea, mode='markers', name='Phase A current'), row=1, col=1)
    fig.add_shape(
    type="line",
    x0=0,
    y0=vara,
    x1=len(phasea) - 1,
    y1=vara,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=1,
    col=1
    )
    
    fig.add_trace(go.Scatter(x=np.arange(len(phaseb)), y=phaseb, mode='markers', name='Phase B current'), row=2, col=1)
    fig.add_shape(
    type="line",
    x0=0,       
    y0=varb,
    x1=len(phaseb) - 1,
    y1=varb,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=2,
    col=1
    )
    
    fig.add_trace(go.Scatter(x=np.arange(len(phasec)), y=phasec, mode='markers', name='Phase C current'), row=3, col=1)
    fig.add_shape(
    type="line",
    x0=0,
    y0=varc,
    x1=len(phasec) - 1,
    y1=varc,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=3,
    col=1
    )
    fig.update_layout(
    height=1200, width=800,
    title="Variance of Phase current Graph"
    )
    fig.update_yaxes(title_text='Phase C current', row=3, col=1)
    fig.update_yaxes(title_text='Phase B current', row=2, col=1)
    fig.update_yaxes(title_text='Phase A current', row=1, col=1)
    for i in range(1, 4):
        fig.update_xaxes(
        range=[0, 100],  # Set the initial range of the x-axis
        row=i, col=1,
        rangeslider=dict(visible=True)  # Enable the rangeslider for each subplot
    )
    st.plotly_chart(fig)
    fig.write_image("figure.png")
else:
    st.text("The signal to noise ratio of the phase A current is " + str(snra))
    st.text("The signal to noise ratio of the phase B current is " + str(snrb))
    st.text("The signal to noise ratio of the phase C current is " + str(snrc))
    
    fig = make_subplots(rows=3, cols=1,vertical_spacing=0.3)
    fig.add_trace(go.Scatter(x=np.arange(len(phasea)), y=phasea, mode='markers', name='Phase A current'), row=1, col=1)
    fig.add_shape(
    type="line",
    x0=0,
    y0=snra,
    x1=len(phasea) - 1,
    y1=snra,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=1,
    col=1
    )
    
    fig.add_trace(go.Scatter(x=np.arange(len(phaseb)), y=phaseb, mode='markers', name='Phase B current'), row=2, col=1)
    fig.add_shape(
    type="line",
    x0=0,       
    y0=snrb,
    x1=len(phaseb) - 1,
    y1=snrb,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=2,
    col=1
    )
    
    fig.add_trace(go.Scatter(x=np.arange(len(phasec)), y=phasec, mode='markers', name='Phase C current'), row=3, col=1)
    fig.add_shape(
    type="line",
    x0=0,
    y0=snrc,
    x1=len(phasec) - 1,
    y1=snrc,
    line=dict(color='red', dash='dash'),
    name='Mean',
    row=3,
    col=1
    )
    fig.update_layout(
    height=1200, width=800,
    title="Signal to Noise ratio of Phase current Graph"
    )
    fig.update_yaxes(title_text='Phase C current', row=3, col=1)
    fig.update_yaxes(title_text='Phase B current', row=2, col=1)
    fig.update_yaxes(title_text='Phase A current', row=1, col=1)
    for i in range(1, 4):
        fig.update_xaxes(
        range=[0, 100],  # Set the initial range of the x-axis
        row=i, col=1,
        rangeslider=dict(visible=True)  # Enable the rangeslider for each subplot
    )
    st.plotly_chart(fig)
    fig.write_image("figure.png")



# result_1.columns=['Phase A Current', 'Phase B Current', 'Phase C Current']

# # Plot the waveform data
# fig, ax = plt.subplots()
# ax.plot(result_1.index, result_1['Phase A Current'], label='Phase A')
# ax.plot(result_1.index, result_1['Phase B Current'], label='Phase B')
# ax.plot(result_1.index, result_1['Phase C Current'], label='Phase C')
# ax.set_xlabel('Sample Index')
# ax.set_ylabel('Current')
# ax.set_title('Input Waveform')
# ax.legend()

# # Display the plot in Streamlit
# st.pyplot(fig)



    
