from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

from datetime import datetime
import pytz
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude,
            dropoff_longitude,dropoff_latitude, passenger_count):
    #On appelle le modèle
    model =joblib.load('model.joblib')

    #On crée une valeur pour la colonne key qui correspond à la date et heure de la demande
    now = datetime.now()

    #create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # now = datetime.strptime(now, "%Y-%m-%d %H:%M:%S") pas besoin c'est déjà une datetime

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    localized_now = eastern.localize(now, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    utc_now = localized_now.astimezone(pytz.utc)

    data = {
        'key': utc_now,
        'pickup_datetime': utc_pickup_datetime,
        'pickup_longitude': float(pickup_longitude),
        'pickup_latitude': float(pickup_latitude),
        'dropoff_longitude': float(dropoff_longitude),
        'dropoff_latitude': float(dropoff_latitude),
        'passenger_count': int(passenger_count)
    }
    query_df = pd.DataFrame(data,index=[0])
    pred=model.predict(query_df)[0]
    # pred=float(pickup_longitude)*float(pickup_latitude)

    return {'fare':pred}
