# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 18:48:54 2022

@author: harik
"""

import numpy as np
from flask import Flask,request,render_template
import pickle
import bz2

#function to decompress and unpickle the data model
def decompress_pickle(file):
   data = bz2.BZ2File(file, 'rb')
   data = pickle.load(data)
   return data

#creating flask app
flask_app = Flask(__name__)

#model=pickle.load(open('model.pkl','rb'))
model = decompress_pickle('model.pbz2')
le = pickle.load(open('label_enc.pkl','rb'))


#print(le.inverse_transform([18]))


@flask_app.route('/')
def home():
    return render_template('home.html')

@flask_app.route('/prediction',methods=['POST'])
def predict():
    PM25= float(request.values['PM2.5'])
    PM25=np.reshape(PM25,(-1,1))
    
    PM10= float(request.values['PM10'])
    PM10=np.reshape(PM10,(-1,1))
    
    NO= float(request.values['NO'])
    NO=np.reshape(NO,(-1,1))
    
    NO2= float(request.values['NO2'])
    NO2=np.reshape(NO2,(-1,1))
    
    NH3= float(request.values['NH3'])
    NH3=np.reshape(NH3,(-1,1))
    
    CO= float(request.values['CO'])
    CO=np.reshape(CO,(-1,1))
    
    SO2= float(request.values['SO2'])
    SO2=np.reshape(SO2,(-1,1))
    
    O3= float(request.values['O3'])
    O3=np.reshape(O3,(-1,1))
    
    Benz= float(request.values['Benzene'])
    Benz=np.reshape(Benz,(-1,1))
    
    Tol= float(request.values['Toluene'])
    Tol=np.reshape(Tol,(-1,1))
    
    Xyl= float(request.values['Xylene'])
    Xyl=np.reshape(Xyl,(-1,1))
    
    City = request.form.get('City')
    City = le.transform([City])  #inverse_transform
    City = City.item()
    print(City)

    features = [City,PM25,PM10,NO,NO2,NH3,CO,SO2,O3,Benz,Tol,Xyl]
    #features = [15.05,43.47,3.30,14.59,19.91,0.95,10.12,22.85,0.10,0.04,1.270,2]
    data_out=np.array(features).reshape(1,-1)
    prediction=model.predict(data_out)
    prediction= prediction.astype('int')
    prediction = prediction.item()
    

    if (prediction >=0) & (prediction <=50):
        bucket = "Good"
    elif (prediction >=51) & (prediction <=100):
         bucket = "Satisfactory"
    elif (prediction >=101) & (prediction <=200):
         bucket = "Moderate"
    elif (prediction >=201) & (prediction <=300):
         bucket = "Poor"
    elif (prediction >=301) & (prediction <=400):
         bucket = "Very Poor"
    else: 
         bucket = "Severe"

        
        
    return render_template('home.html',prediction_text="Air quality index (AQI) is :  {}".format(prediction)
                           ,bucket_text="Air quality bucket (AQI) is : {}".format(bucket))

if __name__=='__main__':
    flask_app.run(port=8000)


