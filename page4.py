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
import pywt
import pyhht
from skimage.restoration import denoise_tv_chambolle
from scipy.signal import firwin, convolve
import psycopg2


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

select_query_stmnt1 = text("SELECT * FROM input LIMIT 200 ;") #input waveform

result_1 = db.execute_dql_commands(select_query_stmnt1)

result_1 = pd.DataFrame(result_1)

select_query_stmnt2 = text("SELECT * FROM output LIMIT 200 ;") #output waveform

result_2 = db.execute_dql_commands(select_query_stmnt2)

result_2 = pd.DataFrame(result_2)

st.write("The input database is -")
st.dataframe(result_1)

st.write("The output database is -")
st.dataframe(result_2)

# opt=["Input Graph", "Output Graph"]
# st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select what statistical feature you want to calculate** :</span>", unsafe_allow_html=True)
# vis_selected=st.selectbox("", options=opt)

def perform_denoising(pure_signal, noisy_signal, wavelet, level, threshold):
    # Perform wavelet decomposition on the noisy signal
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)

    # Apply soft thresholding to each detail coefficient
    denoised_coeffs = []
    for i in range(1, len(coeffs)):
        thresholded_coeffs = pywt.threshold(coeffs[i], threshold, mode='soft')
        denoised_coeffs.append(thresholded_coeffs)

    # Reconstruct the denoised signal using inverse wavelet transform
    denoised_signal = pywt.waverec([coeffs[0]] + denoised_coeffs, wavelet)

    # Return the denoised signal
    return denoised_signal

def cosine_filter(data, cutoff_freq, sample_rate): #cosine signal denoising
    # Normalize the cutoff frequency
    normalized_cutoff = cutoff_freq / (sample_rate / 2)


    # Calculate the filter order
    filter_order = int(10 * sample_rate / cutoff_freq)


    # Generate the filter coefficients using the cosine window
    filter_coeffs = firwin(filter_order, normalized_cutoff, window='cosine')


    # Apply the filter to the data
    filtered_data = convolve(data, filter_coeffs, mode='same')


    return filtered_data

# Given pure and noisy data for each phase current
t = np.linspace(0, 0.02, 240)

# Pure current data for each phase-output current values
pure_current_phase_A = result_2["Phase_A_current"].values
pure_current_phase_B = result_2["Phase_B_current"].values
pure_current_phase_C = result_2["Phase_C_current"].values

# Noisy current data for each phase-input current values
noisy_current_phase_A = result_1["Phase_A_current"].values
noisy_current_phase_B = result_1["Phase_B_current"].values
noisy_current_phase_C = result_1["Phase_C_current"].values

denoising_opt=["Wavelet denoising","EMD denoising","Cosine filter denoising"]
denoising_select=st.selectbox("",options=denoising_opt)

if(denoising_select=="Wavelet denoising"):
    wavelet = 'bior6.8'
    level = 2
    threshold = np.sqrt(2 * np.log(len(noisy_current_phase_A)))

    denoised_current_phase_A = perform_denoising(pure_current_phase_A, noisy_current_phase_A, wavelet, level, threshold)
    denoised_current_phase_B = perform_denoising(pure_current_phase_B, noisy_current_phase_B, wavelet, level, threshold)
    denoised_current_phase_C = perform_denoising(pure_current_phase_C, noisy_current_phase_C, wavelet, level, threshold)

    st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select the denoised current** :</span>", unsafe_allow_html=True)
    curr_opt=["Phase Current A", "Phase Current B", "Phase Current C"];
    selected=st.selectbox("",options=curr_opt)

    if(selected=="Phase Current A"):
        fig=make_subplots(rows=3, cols=1, vertical_spacing=0.2)
        fig.add_trace(go.Scatter(x=t, y=pure_current_phase_A, mode='markers', name=' Pure Phase A current'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=noisy_current_phase_A, mode='markers', name=' Noisy Phase A current'), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=denoised_current_phase_A, mode='markers', name=' Denoised Phase A current'), row=3, col=1)
        fig.update_layout(
        height=1500, width=800,
        title="Denoising of Phase A current"
        )
    elif(selected=="Phase Current B"):
        fig=make_subplots(rows=3, cols=1, vertical_spacing=0.2)
        fig.add_trace(go.Scatter(x=t, y=pure_current_phase_B, mode='markers', name=' Pure Phase B current'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=noisy_current_phase_B, mode='markers', name=' Noisy Phase B current'), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=denoised_current_phase_B, mode='markers', name=' Denoised Phase B current'), row=3, col=1)
        fig.update_layout(
        height=1500, width=800,
        title="Denoising of Phase B current"
        )
    elif(selected=="Phase Current C"):
        fig=make_subplots(rows=3, cols=1, vertical_spacing=0.2)
        fig.add_trace(go.Scatter(x=t, y=pure_current_phase_C, mode='markers', name=' Pure Phase C current'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=noisy_current_phase_C, mode='markers', name=' Noisy Phase C current'), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=denoised_current_phase_C, mode='markers', name=' Denoised Phase C current'), row=3, col=1)
        fig.update_layout(
        height=1500, width=800,
        title="Denoising of Phase C current"
        )
        

    
    fig.update_layout(
    height=1200, width=800,
    title="Visualisation of Phase current Graphs"
    )
    st.plotly_chart(fig)
elif(denoising_select=="EMD denoising"):
    # Perform EMD and denoising for current1
    decomposer = pyhht.EMD(noisy_current_phase_A)
    IMFs = decomposer.decompose()
    denoised_IMFs = [denoise_tv_chambolle(IMFs[i]) for i in range(len(IMFs))]
    denoised_signalA = sum(denoised_IMFs)

    # Perform EMD and denoising for current2
    decomposer = pyhht.EMD(noisy_current_phase_B)
    IMFs = decomposer.decompose()
    denoised_IMFs = [denoise_tv_chambolle(IMFs[i]) for i in range(len(IMFs))]
    denoised_signalB = sum(denoised_IMFs)

    # Perform EMD and denoising for current3
    decomposer = pyhht.EMD(noisy_current_phase_C)
    IMFs = decomposer.decompose()
    denoised_IMFs = [denoise_tv_chambolle(IMFs[i]) for i in range(len(IMFs))]
    denoised_signalC = sum(denoised_IMFs)
    st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select the denoised current** :</span>", unsafe_allow_html=True)
    curr_opt=["Phase Current A", "Phase Current B", "Phase Current C"];
    selected=st.selectbox("",options=curr_opt)
    if(selected=="Phase Current A"):
        fig1 = go.Figure(data=go.Scatter(x=np.arange(len(denoised_signalA)), y=denoised_signalA,mode='markers'))
        fig1.update_layout(width=800, height=500,title="EMD Denoised Current A")
        st.plotly_chart(fig1)
    elif(selected=="Phase Current B"):
        fig2 = go.Figure(data=go.Scatter(x=np.arange(len(denoised_signalB)), y=denoised_signalB,mode='markers'))
        fig2.update_layout(width=800, height=500,title="EMD Denoised Current B")
        st.plotly_chart(fig2)
    elif(selected=="Phase Current C"):
        fig3 = go.Figure(data=go.Scatter(x=np.arange(len(denoised_signalC)), y=denoised_signalC, mode='markers'))
        fig3.update_layout(width=800, height=500,title="EMD Denoised Current C")
        st.plotly_chart(fig3)
elif(denoising_select=="Cosine filter denoising"):
    sample_rate = 4000  # Replace with your sample rate in Hz
    cutoff_freq = 500  # Replace with your desired cutoff frequency in Hz
    denoisedA=cosine_filter(noisy_current_phase_A,cutoff_freq,sample_rate)
    denoisedB=cosine_filter(noisy_current_phase_B,cutoff_freq,sample_rate)
    denoisedC=cosine_filter(noisy_current_phase_C,cutoff_freq,sample_rate)
    st.write("<span style='font-size: 24px; font-family: Times New Roman;'>**Please select the denoised current** :</span>", unsafe_allow_html=True)
    curr_opt=["Phase Current A", "Phase Current B", "Phase Current C"];
    selected=st.selectbox("",options=curr_opt)
    if(selected=="Phase Current A"):
        fig1 = go.Figure(data=go.Scatter(x=np.arange(len(denoisedA)), y=denoisedA,mode='markers'))
        fig1.update_layout(width=800, height=500,title="Cosine Denoised Current A")
        st.plotly_chart(fig1)
    elif(selected=="Phase Current B"):
        fig2 = go.Figure(data=go.Scatter(x=np.arange(len(denoisedB)), y=denoisedB,mode='markers'))
        fig2.update_layout(width=800, height=500,title="Cosine Denoised Current B")
        st.plotly_chart(fig2)
    elif(selected=="Phase Current C"):
        fig3 = go.Figure(data=go.Scatter(x=np.arange(len(denoisedC)), y=denoisedC, mode='markers'))
        fig3.update_layout(width=800, height=500,title="Cosine Denoised Current C")
        st.plotly_chart(fig3)
    
    

def insertvalue():
    conn = psycopg2.connect(
    host=HOST,
    database=DATABASE_NAME,
    user=USER_NAME,
    password=PASSWORD
    )

    # Create a cursor object to execute SQL queries
    cur = conn.cursor()

    # Define your array data
    den_size=len(denoised_current_phase_A);
    timestamp = list(range(1, den_size+1))


    # Define the SQL query with placeholders for the array values
    sq1="DELETE FROM denoised"
    cur.execute(sq1)
    sql = " INSERT INTO denoised (currenta, currentb, currentc, t) VALUES (%s, %s, %s, %s)"

    # Create a list of tuples containing the values to be inserted
    values = [(x, y, z, w) for x, y, z, w in zip(denoised_current_phase_A, denoised_current_phase_B, denoised_current_phase_C,timestamp)]

    # Execute the SQL query for each tuple in the values list
    cur.executemany(sql, values)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and database connection
    cur.close()
    conn.close()
st.write("Click here to update the table with new values")
if st.button('Update'):
    insertvalue()

        
          
    
    


    
