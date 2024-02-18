import csv
import os

import numpy as np
import polars as pl
from fastapi import FastAPI

app = FastAPI()

## A simple FastAPI powered endpoint to take weather observation data from an 
##  Ambient weather WS2902 weather station and:
##    Compute the dewpoint 
##    Generate a new row_id
##    Write the data to a CSV
##  Roughly based on some of the information here: https://ambientweather.com/faqs/question/view/id/1857/
##    and then tweaked to make it work :)

observations_dir = r"/home/aaron/weather_assets/observations"

WEATHER_OBSERVATIONS_CSV_PATH = os.path.join(
    observations_dir, "WEATHER_OBSERVATION_dp.csv"
)
WEATHER_DATA_SCHEMA = {
    "id": pl.Float64,
    "dateutc": pl.Utf8,
    "tempinf": pl.Float64,
    "humidityin": pl.Float64,
    "baromrelin": pl.Float64,
    "baromabsin": pl.Float64,
    "tempf": pl.Float64,
    "humidity": pl.Float64,
    "winddir": pl.Float64,
    "windspeedmph": pl.Float64,
    "windgustmph": pl.Float64,
    "maxdailygust": pl.Float64,
    "hourlyrainin": pl.Float64,
    "eventrainin": pl.Float64,
    "dailyrainin": pl.Float64,
    "weeklyrainin": pl.Float64,
    "monthlyrainin": pl.Float64,
    "totalrainin": pl.Float64,
    "solarradiation": pl.Float64,
    "uv": pl.Float64,
}

def get_dew_point_f(t_air_f, rel_humidity):
    """Compute the dew point in degrees Fahrenheit, modified from https://gist.github.com/sourceperl/45587ea99ff123745428
    :param t_air_f: current ambient temperature in degrees Fahrenheit
    :type t_air_f: float
    :param rel_humidity: relative humidity in %
    :type rel_humidity: float
    :return: the dew point in degrees Fahrenheit
    :rtype: float
    """
    t_air_c = (t_air_f - 32) * 5 / 9

    A = 17.27
    B = 237.7
    alpha = ((A * t_air_c) / (B + t_air_c)) + np.log(rel_humidity / 100.0)
    dp_c = (B * alpha) / (A - alpha)
    dp_f = (dp_c * 9 / 5) + 32
    return dp_f


@app.get("/addWeatherObservation")
async def add_observation(
    stationtype: str,
    PASSKEY: str,
    dateutc: str,
    tempinf: float,
    humidityin: float,
    baromrelin: float,
    baromabsin: float,
    tempf: float,
    humidity: float,
    winddir: float,
    windspeedmph: float,
    windgustmph: float,
    maxdailygust: float,
    hourlyrainin: float,
    eventrainin: float,
    dailyrainin: float,
    weeklyrainin: float,
    monthlyrainin: float,
    totalrainin: float,
    solarradiation: float,
    uv: float,
    batt_co2: float,
):
    ## Calc dewpoint
    dew_point = get_dew_point_f(tempf, humidity)
    
    ## Find max row_id and add one to make a new id.
    row_id = (
        pl.scan_csv(WEATHER_OBSERVATIONS_CSV_PATH, schema=WEATHER_DATA_SCHEMA)
        .select(["id"])
        .max()
        .collect()["id"][0]
        + 1
    )
    
    ## Add new row to CSV
    with open(WEATHER_OBSERVATIONS_CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                row_id,
                dateutc,
                tempinf,
                humidityin,
                baromrelin,
                baromabsin,
                tempf,
                humidity,
                dew_point,
                winddir,
                windspeedmph,
                windgustmph,
                maxdailygust,
                hourlyrainin,
                eventrainin,
                dailyrainin,
                weeklyrainin,
                monthlyrainin,
                totalrainin,
                solarradiation,
                uv,
            ]
        )
    return {"dateutc": dateutc}
