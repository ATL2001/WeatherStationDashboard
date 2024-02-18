#!/usr/bin/env python
# coding: utf-8

import datetime
import logging
import os

import polars as pl
import requests

logname = r"/home/aaron/weather_assets/logs/forecast.log"
logging.basicConfig(
    filename=logname,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

forecast_dir = r"/home/aaron/weather_assets/forecast"
NWS_office = "LSX"
NWS_grid = "103,81"
NWS_headers = {"User-Agent": "i_like_forecasts!", "From": ""}
NWS_forecast_endpoint = (
    f"https://api.weather.gov/gridpoints/{NWS_office}/{NWS_grid}/forecast/hourly"
)

## CSV paths
FORECAST_PREDICTIONS_CSV_PATH = os.path.join(forecast_dir, "FORECAST_PREDICTIONS.csv")
FORECAST_DESCRIPTIONS_CSV_PATH = os.path.join(forecast_dir, "FORECAST_DESCRIPTIONS.csv")
FORECAST_ICONS_CSV_PATH = os.path.join(forecast_dir, "FORECAST_ICONS.csv")

## Dictionaries of constants
FORECAST_DESCRIPTIONS_SCHEMA = {
    "id": pl.Int64,
    "shortForecast": pl.Utf8,
}

FORECAST_ICONS_SCHEMA = {
    "id": pl.Int64,
    "icon": pl.Utf8,
}

NWS_REQUEST_SCHEMA = {
    "number": pl.Int16,
    "startTime": pl.Utf8,
    "endTime": pl.Utf8,
    "isDaytime": bool,
    "temperature": pl.Int16,  # Int8 should be safe, but whatever...
    "probabilityOfPrecipitation": pl.Unknown,
    "dewpoint": pl.Unknown,
    "windSpeed": pl.Utf8,
    "windDirection": pl.Utf8,
    "icon": pl.Utf8,
    "shortForecast": pl.Utf8,
}

FORECAST_PREDICTIONS_SCHEMA = {
    "id": pl.Float64,
    "number": pl.Int64,
    "startTime": pl.Datetime("us", "UTC"),
    "endTime": pl.Datetime("us", "UTC"),
    "isDaytime": bool,
    "temperature": pl.Int16,
    "probabilityOfPrecipitation": pl.Int8,
    "dewpoint": pl.Int16,
    "windSpeed": pl.Int16,
    "windDirection": pl.Int16,
    "forecast_updated_time": pl.Datetime("us", "UTC"),
    "forecast_descriptions_id": pl.Int64,
    "icon_id": pl.Int64,
}

WIND_DIR_REPLACE_DICT = {
    "N": 0,
    "NW": 315,
    "W": 270,
    "SW": 225,
    "S": 180,
    "SE": 135,
    "E": 90,
    "NE": 45,
}


#### Forecast functions
def make_forecast_descriptions_lf() -> pl.LazyFrame:
    if os.path.exists(FORECAST_DESCRIPTIONS_CSV_PATH):
        forecast_descriptions_lf = pl.scan_csv(FORECAST_DESCRIPTIONS_CSV_PATH)
    else:
        forecast_descriptions_lf = pl.LazyFrame([], schema=FORECAST_DESCRIPTIONS_SCHEMA)
    return forecast_descriptions_lf


def make_forecast_icons_lf() -> pl.LazyFrame:
    if os.path.exists(FORECAST_ICONS_CSV_PATH):
        forecast_icons_lf = pl.scan_csv(FORECAST_ICONS_CSV_PATH)
    else:
        forecast_icons_lf = pl.LazyFrame([], schema=FORECAST_ICONS_SCHEMA)
    return forecast_icons_lf


def add_new_forecast_descriptions(NWS_df, forecast_descriptions_lf) -> bool:
    if (
        len(
            NWS_df.filter(pl.col("forecast_descriptions_id").is_null())[
                "shortForecast"
            ].unique()
        )
        > 0
    ):
        max_description_id = forecast_descriptions_lf.collect()["id"].max()
        if max_description_id is None:
            max_description_id = 0

        new_forecast_descriptions_df = (
            (
                pl.LazyFrame(
                    NWS_df.filter(pl.col("forecast_descriptions_id").is_null())[
                        "shortForecast"
                    ].unique()
                )
                .with_row_count()
                .with_columns(
                    (pl.col("row_nr") + max_description_id).cast(pl.Int64).alias("id")
                )
                .drop("row_nr")
            )
            .collect()
            .select([pl.col("id"), pl.col("shortForecast")])
        )
        forecast_descriptions_df = pl.concat(
            [forecast_descriptions_lf.collect(), new_forecast_descriptions_df]
        )
        forecast_descriptions_df.write_csv(FORECAST_DESCRIPTIONS_CSV_PATH)
        return True
    else:
        return False


def add_new_forecast_icons(NWS_df, forecast_icons_lf) -> bool:
    if len(NWS_df.filter(pl.col("icon_id").is_null())["icon"].unique()) > 0:
        max_icon_id = forecast_icons_lf.collect()["id"].max()
        if max_icon_id is None:
            max_icon_id = 0

        new_forecast_icons_df = (
            (
                pl.LazyFrame(
                    NWS_df.filter(pl.col("icon_id").is_null())["icon"].unique()
                )
                .with_row_count()
                .with_columns(
                    (pl.col("row_nr") + max_icon_id).cast(pl.Int64).alias("id")
                )
                .drop("row_nr")
            )
            .collect()
            .select([pl.col("id"), pl.col("icon")])
        )
        new_forecast_icons_df = pl.concat(
            [forecast_icons_lf.collect(), new_forecast_icons_df]
        )
        new_forecast_icons_df.write_csv(FORECAST_ICONS_CSV_PATH)
        return True
    else:
        return False


def make_forecast_frames(NWS_r: dict) -> pl.LazyFrame:
    forecast_descriptions_lf = make_forecast_descriptions_lf()
    forecast_icons_lf = make_forecast_icons_lf()
    NWS_df = (
        ## create lazyframe of NWS forecast data from the response
        pl.DataFrame(
            ## get the periods of the NWS response, list of dictionaries for forecast information
            NWS_r.json()["properties"]["periods"],
            schema=NWS_REQUEST_SCHEMA,
        )
        .lazy()
        ## unnest probabilityOfPrecipitation dictionary into into columns for
        ##  its unitCode and value keys  {'unitCode': 'wmoUnit:percent', 'value': 0},
        .unnest("probabilityOfPrecipitation")
        ## unit code from probabilityOfPrecipitation, always percent, drop it
        .drop("unitCode")
        ## value code from probabilityOfPrecipitation, rename it
        .rename({"value": "probabilityOfPrecipitation"})
        ## unnest dewpoint dictionary into columns for its
        ##  unitCode and value keys {'unitCode': 'wmoUnit:degC', 'value': 15.555555555555555}
        .unnest("dewpoint")
        ## unit code from dewpoint, always °C, drop it
        .drop("unitCode")
        ## value code from dewpoint, rename it
        .rename({"value": "dewpoint"})
        ## transform NWS data to how we want it
        .with_columns(
            ## transform starttime string to datetime
            pl.col("startTime").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False
            ),
            ## transform endtime string to datetime
            pl.col("endTime").str.strptime(
                pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False
            ),
            ## split the windspeed string at the space and get the first group and cast it as an int
            ##   so strings like "12 MPH" just turn into 12
            pl.col("windSpeed").str.split(" ").list.first().cast(pl.Int16),
            ## replace the wind direction strings with degrees
            pl.col("windDirection").map_dict(WIND_DIR_REPLACE_DICT).cast(pl.Int16),
            ## add a column for the forecast updated datetime with the updated property of the NWS response
            pl.lit(NWS_r.json()["properties"]["updated"])
            .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False)
            .alias("forecast_updated_time"),
            ## convert dewpoint to °F
            (pl.col("dewpoint") * 9 / 5 + 32).cast(pl.Int16),
            ## convert probabilityOfPrecipitation to int8
            pl.col("probabilityOfPrecipitation").cast(pl.Int8),
        )
        ## join in the forecast descriptions
        .join(forecast_descriptions_lf, on="shortForecast", how="left")
        ## rename id to keep it straight
        .rename({"id": "forecast_descriptions_id"})
        ## join in the forecast descriptions
        .join(forecast_icons_lf, on="icon", how="left")
        ## rename id to keep it straight
        .rename({"id": "icon_id"})
    ).collect()
    return NWS_df, forecast_descriptions_lf, forecast_icons_lf


def make_existing_predictions_lf() -> pl.LazyFrame:
    if os.path.exists(FORECAST_PREDICTIONS_CSV_PATH):
        forecast_predictions_lf = pl.scan_csv(
            FORECAST_PREDICTIONS_CSV_PATH,
            try_parse_dates=True,
            dtypes=FORECAST_PREDICTIONS_SCHEMA,
        )
    else:
        forecast_predictions_lf = pl.LazyFrame([], schema=FORECAST_PREDICTIONS_SCHEMA)
    return forecast_predictions_lf


def add_new_predictions(new_NWS_df) -> bool:
    try:
        forecast_predictions_lf = make_existing_predictions_lf()
        max_prediction_id = forecast_predictions_lf.collect()["id"].max()
        if max_prediction_id is None:
            max_prediction_id = 0
        new_NWS_df = new_NWS_df.with_columns(pl.col("number").cast(pl.Int64))
        new_NWS_df = new_NWS_df.with_columns(
            (pl.col("number") + max_prediction_id).cast(pl.Float64).alias("id"),
        ).select(
            [
                pl.col("id"),
                pl.col("number"),
                pl.col("startTime"),
                pl.col("endTime"),
                pl.col("isDaytime"),
                pl.col("temperature"),
                pl.col("probabilityOfPrecipitation"),
                pl.col("dewpoint"),
                pl.col("windSpeed"),
                pl.col("windDirection"),
                pl.col("forecast_updated_time"),
                pl.col("forecast_descriptions_id"),
                pl.col("icon_id"),
            ]
        )
        new_NWS_df = pl.concat([forecast_predictions_lf.collect(), new_NWS_df])
        new_NWS_df.write_csv(FORECAST_PREDICTIONS_CSV_PATH)
        return True
    except Exception as ex:
        print(f"Exception occured while adding new predictions to csv.\n{ex}")
        return False


try:
    # request the forecast from NWS
    NWS_r = requests.get(NWS_forecast_endpoint, headers=NWS_headers)

    if NWS_r.status_code == 200:
        ## compare the updated time of the forecast to the max forecast update time in the csv
        forecast_updated_time = datetime.datetime.fromisoformat(
            NWS_r.json()["properties"]["updated"]
        )
        max_forecast_updated_time_in_CSV = (
            make_existing_predictions_lf().collect()["forecast_updated_time"].max()
        )
        if max_forecast_updated_time_in_CSV is None:
            max_forecast_updated_time_in_CSV = (
                datetime.datetime.utcnow() - datetime.timedelta(365)
            )
            max_forecast_updated_time_in_CSV = max_forecast_updated_time_in_CSV.replace(
                tzinfo=datetime.timezone.utc
            )
        if forecast_updated_time <= max_forecast_updated_time_in_CSV:
            logging.info(
                f"CSV already up to date with the most recent NWS forecast data. {forecast_updated_time}"
            )
        else:
            ## Make the NWS df and the description/icons lazy frames
            (
                new_NWS_df,
                forecast_descriptions_lf,
                forecast_icons_lf,
            ) = make_forecast_frames(NWS_r)

            ## add any new icons or descriptions to the appropriate csvs
            added_forecast_descriptions = add_new_forecast_descriptions(
                new_NWS_df, forecast_descriptions_lf
            )
            added_icons = add_new_forecast_icons(new_NWS_df, forecast_icons_lf)

            if added_forecast_descriptions or added_icons:
                ## if we added new descriptions or icons, we dont have a good id for those rows
                ##  we can just rerun the makeframes againand get the new ids
                (
                    new_NWS_df,
                    forecast_descriptions_lf,
                    forecast_icons_lf,
                ) = make_forecast_frames(NWS_r)

            new_NWS_df = new_NWS_df.drop(["icon", "shortForecast"])
            if add_new_predictions(new_NWS_df):
                logging.info("Successfully added new predictions to csv.")

    else:
        logging.warning(f"{NWS_r.status_code=}")
        logging.warning(NWS_r.content)
        ## status of NWS forecast was not 200, service was down or something.
        ## perhaps do something else here for NWS being down, it happens...
        pass
except Exception as ex:
    logging.error(f"Something died getting weather data from nws. {ex}")
