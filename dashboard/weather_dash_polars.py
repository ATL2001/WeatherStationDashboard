#!/usr/bin/env python
# coding: utf-8

import base64
import datetime
import logging
import os
import warnings
from typing import List

import astral
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import flask
import plotly
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import scipy
from astral.location import Location
from dash import Dash, Input, Output, dcc, html
from dash_bootstrap_templates import load_figure_template

logname = r"/home/aaron/weather_assets/logs/dashboard.log"
logging.basicConfig(
    filename=logname,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logging.info("Dashboard loading.")

radar_img_download_path = r"/home/aaron/weather_assets/dashboard/assets/radar.gif"

home_coordinates = (38.78296, -89.93201)
home_town = "Edwardsville"
home_country = "USA"
timezone_string = "US/Central"

forecast_dir = r"/home/aaron/weather_assets/forecast"

## CSV paths
WEATHER_OBSERVATIONS_CSV_PATH = (
    r"/home/aaron/weather_assets/observations/WEATHER_OBSERVATION_dp.csv"
)
FORECAST_PREDICTIONS_CSV_PATH = os.path.join(forecast_dir, "FORECAST_PREDICTIONS.csv")
FORECAST_DESCRIPTIONS_CSV_PATH = os.path.join(forecast_dir, "FORECAST_DESCRIPTIONS.csv")
FORECAST_ICONS_CSV_PATH = os.path.join(forecast_dir, "FORECAST_ICONS.csv")

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
FORECAST_PREDICTIONS_SCHEMA = {
    "id": pl.Float64,
    "number": pl.Int64,
    "startTime": pl.Datetime("ns", "UTC"),
    "endTime": pl.Datetime("ns", "UTC"),
    "isDaytime": bool,
    "temperature": pl.Float64,
    "probabilityOfPrecipitation": pl.Int8,
    "dewpoint": pl.Float64,
    "windSpeed": pl.Float64,
    "windDirection": pl.Float64,
    "forecast_updated_time": pl.Datetime("ns", "UTC"),
    "forecast_descriptions_id": pl.Int64,
    "icon_id": pl.Int64,
}

WEATHER_DF_SCHEMA = {
    "DATE": pl.Utf8,  
    "TEMP": pl.Float64,
    "DEWPOINT": pl.Float64,
    "WIND_SPEED": pl.Int16,
    "WIND_DIRECTION": pl.Int16,
    "GUST_SPEED": pl.Int16,
    "RAIN_HOURLY": pl.Float64,
    "RAIN_DAILIY": pl.Float64,
    "RAIN_EVENT": pl.Float64,
    "PRECIP_PROB": pl.Int8,
    "PREDICTION": pl.Boolean,
}

HIGH_LOW_ALIAS_COL_DICT = {
    "High Temp": "TEMP",
    "Low Temp": "TEMP",
    "High Windspeed": "WIND_SPEED",
}
WIND_ANNOTATIONS = ["High Windspeed"]
TEMP_ANNOTATIONS = [
    "High Temp",
    "Low Temp",
]

PLOT_FIG_HEIGHT = 225
PLOT_FIG_WIDTH = 540
PAPER_BG_COLOR = "#222222"

TEMP_PLOT_PADDING = 5  # adds this many degrees to the max and min temps for the upper and lower bounds of the plot
WIND_PLOT_PADDING = (
    2  # adds this many MPH to the max windspeed for the upper bounds of the plot
)
RAIN_PLOT_PADDING = (
    1  # adds this many inches to the max daily rain for the upper bounds of the plot
)

ROW_TWO_SIZE = 10



plotly.io.json.config.default_engine = "orjson"
astral_location = Location(
    astral.LocationInfo(
        home_town,
        home_country,
        timezone_string,
        home_coordinates[0],
        home_coordinates[1],
    )
)
load_figure_template("darkly")
warnings.filterwarnings(
    "ignore",
    ".*A builtin ctypes object gave a PEP3118.*",
)


def read_weather_observations() -> pl.DataFrame:
    try:
        observation_lf = pl.scan_csv(
            WEATHER_OBSERVATIONS_CSV_PATH, dtypes=WEATHER_DATA_SCHEMA
        )
        observation_df = (
            observation_lf.filter(pl.col("id") >= pl.col("id").max() - 4000)
            .drop(
                [
                    "id",
                    "tempinf",
                    "humidityin",
                    "baromrelin",
                    "baromabsin",
                    "humidity",
                    "maxdailygust",
                    "weeklyrainin",
                    "monthlyrainin",
                    "totalrainin",
                    "solarradiation",
                    "uv",
                ]
            )
            .with_columns(
                ## transform starttime string to datetime
                pl.col("dateutc")
                .str.strptime(
                    pl.Datetime,
                    format="%Y-%m-%d %H:%M:%S",
                    strict=False,
                )
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(timezone_string)
                .alias("datect")
                .dt.replace_time_zone(None)
                .cast(pl.Datetime("ns"))
            )
            .filter(
                pl.col("datect")
                > datetime.datetime.utcnow() - datetime.timedelta(days=3)
            )
            .drop(["dateutc"])
            .sort(pl.col("datect"), descending=True)
            .rename(
                {
                    "tempf": "TEMP",
                    "winddir": "WIND_DIRECTION",
                    "windspeedmph": "WIND_SPEED",
                    "windgustmph": "GUST_SPEED",
                    "hourlyrainin": "RAIN_HOURLY",
                    "eventrainin": "RAIN_EVENT",
                    "dailyrainin": "RAIN_DAILIY",
                    "dewpointf": "DEWPOINT",
                    "datect": "DATE",
                }
            )
            .select(
                [
                    "DATE",
                    "TEMP",
                    pl.col("DEWPOINT").cast(pl.Float64).round(1),
                    "WIND_SPEED",
                    "WIND_DIRECTION",
                    "GUST_SPEED",
                    "RAIN_HOURLY",
                    "RAIN_DAILIY",
                    "RAIN_EVENT",
                ]
            )
        ).collect()
        return observation_df
    except Exception as ex:
        func_name = "read_weather_observations"
        logging.error(f"Exception in {func_name}: \n{ex}")


def read_weather_predictions() -> pl.DataFrame:
    try:
        predictions_lf = pl.scan_csv(
            FORECAST_PREDICTIONS_CSV_PATH,
            try_parse_dates=True,
            dtypes=FORECAST_PREDICTIONS_SCHEMA,
        )
        forecast_update_date = predictions_lf.select(
            pl.col("forecast_updated_time").max()
        ).collect()["forecast_updated_time"][0]
        predictions_df = (
            predictions_lf.drop(
                [
                    "id",
                    "number",
                    "endTime",
                    "isDaytime",
                    "forecast_descriptions_id",
                    "icon_id",
                ]
            )
            .filter(
                (pl.col("forecast_updated_time") == forecast_update_date)
                & (
                    pl.col("startTime")
                    > datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
                    - datetime.timedelta(minutes=datetime.datetime.utcnow().minute)
                )
            )
            .with_columns(
                pl.col("startTime")
                .dt.convert_time_zone(timezone_string)
                .dt.replace_time_zone(None),
                pl.col("forecast_updated_time")
                .dt.replace_time_zone(timezone_string)
                .dt.replace_time_zone(None),
                pl.lit(True).alias("PREDICTION"),
                pl.col("temperature").cast(pl.Float64),
                pl.col("dewpoint").cast(pl.Float64),
            )
            .sort("startTime")
            .rename(
                {
                    "startTime": "DATE",
                    "temperature": "TEMP",
                    "probabilityOfPrecipitation": "PRECIP_PROB",
                    "dewpoint": "DEWPOINT",
                    "windSpeed": "WIND_SPEED",
                    "windDirection": "WIND_DIRECTION",
                }
            )
            .select(
                [
                    "DATE",
                    "TEMP",
                    "DEWPOINT",
                    "WIND_SPEED",
                    "WIND_DIRECTION",
                    "PRECIP_PROB",
                    "PREDICTION",
                ]
            )
        ).collect()
        return predictions_df
    except Exception as ex:
        func_name = "read_weather_predictions"
        logging.error(f"Exception in {func_name}: \n{ex}")


def make_weather_df() -> pl.DataFrame:
    try:
        observation_df = read_weather_observations()
        predictions_df = read_weather_predictions()
        weather_df = pl.concat([observation_df, predictions_df], how="diagonal")
        return weather_df
    except Exception as ex:
        func_name = "make_weather_df"
        logging.error(f"Exception in {func_name}: \n{ex}")


def make_high_low_annotations_dict(weather_df: pl.DataFrame) -> dict:
    try:
        high_low_df = weather_df.groupby(pl.col("DATE").dt.day()).agg(
            [
                pl.col("TEMP").max().alias("High Temp"),
                pl.col("TEMP").min().alias("Low Temp"),
                pl.col("WIND_SPEED").max().alias("High Windspeed"),
            ]
        )
        annotations_dict = {}
        for row in high_low_df.iter_rows(named=True):
            for anno_type in HIGH_LOW_ALIAS_COL_DICT.keys():
                if row[anno_type] is not None:
                    this_df = weather_df.filter(
                        (pl.col("DATE").dt.day() == row["DATE"])
                        & (pl.col(HIGH_LOW_ALIAS_COL_DICT[anno_type]) == row[anno_type])
                    ).select("DATE")
                    if anno_type in annotations_dict.keys():
                        annotations_dict[anno_type].append(
                            (this_df["DATE"][0], row[anno_type])
                        )
                    else:
                        annotations_dict[anno_type] = [
                            (this_df["DATE"][0], row[anno_type])
                        ]
        return annotations_dict
    except Exception as ex:
        func_name = "make_high_low_annotations_dict"
        logging.error(f"Exception in {func_name}: \n{ex}")


def query_most_recent_data():
    try:
        observation_lf = pl.scan_csv(
            WEATHER_OBSERVATIONS_CSV_PATH, dtypes=WEATHER_DATA_SCHEMA
        )
        observation_df = (
            observation_lf.filter(pl.col("id") == pl.col("id").max())
        ).collect()

        temp = observation_df["tempf"][0]
        windspeed = observation_df["windspeedmph"][0]
        winddir = observation_df["winddir"][0]
        humidity = observation_df["humidity"][0]
        gust = observation_df["windgustmph"][0]
        pressure = observation_df["baromrelin"][0]
        rain = observation_df["dailyrainin"][0]
        uv = observation_df["uv"][0]
        radiation = observation_df["solarradiation"][0]
        return temp, windspeed, winddir, humidity, gust, pressure, rain, uv, radiation
    except Exception as ex:
        func_name = "query_most_recent_data"
        logging.error(f"Exception in {func_name}: \n{ex}")


def query_high_low():
    try:
        observation_lf = pl.scan_csv(
            WEATHER_OBSERVATIONS_CSV_PATH, dtypes=WEATHER_DATA_SCHEMA
        )
        midnight_today = datetime.datetime.now()
        midnight_today = midnight_today.replace(hour=0, minute=0, second=0)
        observation_df = (
            observation_lf.filter(pl.col("id") >= pl.col("id").max() - 4000)
            .drop(
                [
                    "id",
                    "tempinf",
                    "humidityin",
                    "baromrelin",
                    "baromabsin",
                    "humidity",
                    "maxdailygust",
                    "weeklyrainin",
                    "monthlyrainin",
                    "totalrainin",
                    "solarradiation",
                    "uv",
                ]
            )
            .with_columns(
                ## transform starttime string to datetime
                pl.col("dateutc")
                .str.strptime(
                    pl.Datetime,
                    format="%Y-%m-%d %H:%M:%S",
                    strict=False,
                )
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(timezone_string)
                .alias("datect")
                .dt.replace_time_zone(None)
            )
            .filter(pl.col("datect") >= midnight_today)
            .drop(["dateutc"])
            .select(
                [
                    pl.col("tempf").max().alias("todays_high_temp"),
                    pl.col("tempf").min().alias("todays_low_temp"),
                    pl.col("windspeedmph").max().alias("todays_high_windspeed"),
                ]
            )
        ).collect()
        observation_df
        todays_high_temp = observation_df["todays_high_temp"][0]
        todays_low_temp = observation_df["todays_low_temp"][0]
        todays_high_windspeed = observation_df["todays_high_windspeed"][0]

        return todays_high_temp, todays_low_temp, todays_high_windspeed
    except Exception as ex:
        func_name = "query_high_low"
        logging.error(f"Exception in {func_name}: \n{ex}")


def filter_weather_df_by_date(
    weather_df: pl.DataFrame,
    plot_start_date: datetime.datetime,
    plot_end_date: datetime.datetime,
) -> pl.DataFrame:
    try:
        if weather_df.dtypes[0] != pl.Datetime:
            weather_df = weather_df.with_columns(
                ## transform starttime string to datetime
                pl.col("DATE")
                .str.strptime(
                    pl.Datetime,
                    format="%Y-%m-%dT%H:%M:%S",
                    strict=False,
                )
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(timezone_string)
                .dt.replace_time_zone(None)
            )

        ## filter the weather_df to the start/end dates
        if plot_start_date is None:
            plot_start_date = datetime.datetime.now() - datetime.timedelta(days=1)
            plot_start_date = plot_start_date.replace(
                hour=0,
                minute=0,
                second=1,
            )

        if plot_end_date is None:
            plot_end_date = datetime.datetime.now() + datetime.timedelta(days=1)
            plot_end_date = plot_end_date.replace(
                hour=23, minute=59, second=59, microsecond=9999
            )

        plot_df = weather_df.filter(
            (pl.col("DATE") >= plot_start_date) & (pl.col("DATE") <= plot_end_date)
        )
        return plot_df
    except Exception as ex:
        func_name = "filter_weather_df_by_date"
        logging.error(f"Exception in {func_name}: \n{ex}")


def add_background_shading_to_figure(
    fig: go.Figure, plot_start_date: datetime.datetime, plot_end_date: datetime.datetime
) -> go.Figure:
    try:
        ## add day/night shading
        plot_start_date = plot_start_date.replace(second=0)

        for day_num in range((plot_end_date - plot_start_date).days + 2):
            day = plot_start_date + datetime.timedelta(days=day_num)
            AM_midnight = day.replace(hour=0, minute=0, second=0)
            PM_midnight = day.replace(hour=23, minute=59, second=59)
            sunrise = astral_location.sunrise(day).replace(tzinfo=None)
            sunset = astral_location.sunset(day).replace(tzinfo=None)
            if plot_start_date <= AM_midnight:
                midnight_anno = AM_midnight.strftime("%b %d")
                fig.add_vline(
                    x=AM_midnight.timestamp() * 1000 + 1,
                    line_color="gray",
                    opacity=1,
                    line_width=1,
                    annotation_text=midnight_anno,
                    annotation_font_color="gray",
                    annotation_position="top",
                )
                if plot_end_date >= sunrise:
                    sunrise_anno = sunrise.strftime("%I:%M %p")
                    fig.add_vline(
                        x=sunrise.timestamp() * 1000 + 1,
                        line_color="orange",
                        opacity=0.3,
                        line_width=2,
                        annotation_text=sunrise_anno,
                        annotation_font_color="gray",
                        annotation_position="top",
                    )
                    fig.add_vrect(
                        x0=AM_midnight,
                        x1=sunrise,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
                else:
                    fig.add_vrect(
                        x0=AM_midnight,
                        x1=plot_end_date,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
            else:
                if plot_end_date >= sunrise:
                    sunrise_anno = sunrise.strftime("%I:%M %p")
                    fig.add_vline(
                        x=sunrise.timestamp() * 1000 + 1,
                        line_color="orange",
                        opacity=0.3,
                        line_width=2,
                        annotation_text=sunrise_anno,
                        annotation_font_color="gray",
                        annotation_position="top",
                    )
                    fig.add_vrect(
                        x0=plot_start_date,
                        x1=sunrise,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
                else:
                    fig.add_vrect(
                        x0=plot_start_date,
                        x1=plot_end_date,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
            if plot_start_date <= sunset:
                sunset_anno = sunset.strftime("%I:%M %p")
                fig.add_vline(
                    x=sunset.timestamp() * 1000 + 1,
                    line_color="orange",
                    opacity=0.3,
                    line_width=2,
                    annotation_text=sunset_anno,
                    annotation_font_color="gray",
                    annotation_position="top",
                )
                if plot_end_date >= PM_midnight:
                    fig.add_vrect(
                        x0=sunset,
                        x1=PM_midnight,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
                else:
                    fig.add_vrect(
                        x0=sunset,
                        x1=plot_end_date,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
            else:
                if plot_end_date > PM_midnight:
                    fig.add_vrect(
                        x0=plot_start_date,
                        x1=PM_midnight,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
                else:
                    fig.add_vrect(
                        x0=plot_start_date,
                        x1=plot_end_date,
                        fillcolor="black",
                        opacity=0.3,
                        line_width=0,
                        layer="below",
                    )
        return fig
    except Exception as ex:
        func_name = "add_background_shading_to_figure"
        logging.error(f"Exception in {func_name}: \n{ex}")


def make_temp_fig(
    weather_df: pl.DataFrame,
    plot_start_date: datetime.datetime = None,
    plot_end_date: datetime.datetime = None,
) -> go.Figure:
    try:
        plot_df = filter_weather_df_by_date(weather_df, plot_start_date, plot_end_date)
        plot_start_date = plot_df["DATE"].min()
        plot_end_date = plot_df["DATE"].max()

        ## get the min and max temps for setting the bounds of the plot
        min_temp = weather_df["DEWPOINT"].min() - TEMP_PLOT_PADDING
        max_temp = weather_df["TEMP"].max() + TEMP_PLOT_PADDING

        ## create the figure
        temp_fig = go.Figure(
            layout=go.Layout(
                height=PLOT_FIG_HEIGHT,
                width=PLOT_FIG_WIDTH,
                paper_bgcolor=PAPER_BG_COLOR,
                showlegend=False,
                yaxis=go.layout.YAxis(range=(min_temp, max_temp), title="°F"),
                xaxis=go.layout.XAxis(
                    range=(plot_start_date, plot_end_date),
                    tickformat="%-I:%M %p",
                    rangeslider={"visible": False},
                ),
                margin=go.layout.Margin(autoexpand=True, l=15, r=25, t=20, b=20),
            ),
        )

        temp_fig = add_background_shading_to_figure(
            temp_fig, plot_start_date, plot_end_date
        )

        ## add traces for temp, dewpoint, predicted temp and predicted dewpoint
        temp_fig.add_traces(
            [
                go.Scattergl(
                    x=plot_df.filter(pl.col("PREDICTION").is_null())["DATE"],
                    y=plot_df.filter(pl.col("PREDICTION").is_null())["TEMP"],
                    line=dict(color="rgb(255,0,0)"),
                    mode="lines",
                    name="Temp (°F)",
                    showlegend=True,
                    hovertemplate="Temp: %{y}°<br>%{x|%I:%M %p}<extra></extra>",
                ),
                go.Scattergl(
                    x=plot_df.filter(pl.col("PREDICTION").is_null())["DATE"],
                    y=plot_df.filter(pl.col("PREDICTION").is_null())["DEWPOINT"],
                    line=dict(color="rgb(0,0,255)"),
                    mode="lines",
                    name="Dew (°F)",
                    showlegend=True,
                    hovertemplate="Dewpoint: %{y}°<br>%{x|%I:%M %p}<extra></extra>",
                ),
                go.Scattergl(
                    x=plot_df.filter(pl.col("PREDICTION") == True)["DATE"],  # noqa: E712
                    y=plot_df.filter(pl.col("PREDICTION") == True)["TEMP"],  # noqa: E712
                    line=dict(color="rgb(255,0,0)", dash="dash"),
                    mode="lines",
                    name="Predicted Temp (°F)",
                    showlegend=True,
                    hovertemplate="Predicted Temp: %{y}°<br>%{x|%I:%M %p}<extra></extra>",
                ),
                go.Scattergl(
                    x=plot_df.filter(pl.col("PREDICTION") == True)["DATE"],  # noqa: E712
                    y=plot_df.filter(pl.col("PREDICTION") == True)["DEWPOINT"],  # noqa: E712
                    line=dict(color="rgb(0,0,255)", dash="dash"),
                    mode="lines",
                    name="Predicted Dew (°F)",
                    showlegend=True,
                    hovertemplate="Predicted Dewpoint: %{y}°<br>%{x|%I:%M %p}<extra></extra>",
                ),
            ]
        )

        ## make the high and low annotations dictionary
        annotations_dict = make_high_low_annotations_dict(weather_df)

        ## add highs and lows to the figure
        for anno_type in TEMP_ANNOTATIONS:
            for temp_tuple in annotations_dict[anno_type]:
                time, temp = temp_tuple
                if time > plot_start_date and time <= plot_end_date:
                    if "High" in anno_type:
                        label_position = "bottom center"
                    else:
                        label_position = "top center"
                    temp_fig.add_traces(
                        go.Scattergl(
                            x=[time],
                            y=[temp],
                            mode="markers+text",
                            text=temp,
                            textposition=label_position,
                            textfont={"family": "Arial Black", "size": 12},
                            marker=go.scattergl.Marker(
                                size=8,
                                color="#222222",
                                symbol="circle",
                                line=go.scattergl.marker.Line(color="red", width=2),
                            ),
                        )
                    )

        ## Add 32° line if visible
        if (max_temp > 32) and (min_temp < 32):
            temp_fig.add_hline(
                y=32,
                line_color="dodgerblue",
                opacity=0.95,
                line_width=1,
                annotation_text="32",
                annotation_font_color="dodgerblue",
                annotation_position="left",
            )

        return temp_fig
    except Exception as ex:
        func_name = "make_temp_fig"
        logging.error(f"Exception in {func_name}: \n{ex}")


def make_wind_fig(
    weather_df: pl.DataFrame,
    plot_start_date: datetime.datetime = None,
    plot_end_date: datetime.datetime = None,
) -> go.Figure:
    try:
        plot_df = filter_weather_df_by_date(weather_df, plot_start_date, plot_end_date)

        plot_start_date = plot_df["DATE"].min()
        plot_end_date = plot_df["DATE"].max()

        ## get the number of seconds of the tiem extent,
        ##  then divide by 180 to make the plot not look too crowded
        ##  180 looks like a nice number on the plot in my opinion
        try:
            delta = plot_end_date - plot_start_date
            if delta.total_seconds() > 60 * 15:
                gb_min = int(delta.total_seconds() / 60 / 180)
            else:
                gb_min = 15
        except Exception:
            gb_min = 2
        if gb_min < 2:
            gb_min = 2

        plot_df = (
            plot_df.sort("DATE")
            .groupby_dynamic(pl.col("DATE"), every=f"""{gb_min}m""")
            .agg(
                pl.col("WIND_SPEED").max(),
                pl.col("TEMP").max(),
                pl.col("PREDICTION").max(),
                pl.col("WIND_DIRECTION")
                .radians()
                .cast(pl.Float32)
                .apply(lambda x: scipy.stats.circmean(x))
                .degrees(),
            )
        )
        min_wind = 0
        max_wind = weather_df["WIND_SPEED"].max() + WIND_PLOT_PADDING

        ## create the figure
        wind_fig = go.Figure(
            layout=go.Layout(
                height=PLOT_FIG_HEIGHT,
                width=PLOT_FIG_WIDTH,
                paper_bgcolor=PAPER_BG_COLOR,
                showlegend=False,
                yaxis=go.layout.YAxis(range=(min_wind, max_wind), title="MPH"),
                xaxis=go.layout.XAxis(
                    range=(plot_start_date, plot_end_date),
                    tickformat="%-I:%M %p",
                    rangeslider={"visible": False},
                ),
                margin=go.layout.Margin(autoexpand=True, l=1, r=1, t=20, b=20),
            ),
        )

        wind_fig = add_background_shading_to_figure(
            wind_fig, plot_start_date, plot_end_date
        )

        wind_scatters = []
        for row in plot_df.iter_rows(named=True):
            winddir = row["WIND_DIRECTION"]
            if row["PREDICTION"] is True:
                wind_scatters.append(
                    go.Scattergl(
                        x=(row["DATE"],),
                        y=(row["WIND_SPEED"],),
                        mode="markers",
                        marker={
                            "symbol": "arrow",
                            "color": "chartreuse",
                            "size": 10,
                            "angle": row["WIND_DIRECTION"],
                        },
                        name="Wind speed",
                        showlegend=True,
                        hovertemplate=f"Predicted Wind speed: %{{y}}MPH<br>Wind Direction: {winddir:.0f}<br>%{{x|%I:%M %p}}<extra></extra>",
                    )
                )
            else:
                wind_scatters.append(
                    go.Scattergl(
                        x=(row["DATE"],),
                        y=(row["WIND_SPEED"],),
                        mode="markers",
                        marker={
                            "symbol": "arrow",
                            "color": "dodgerblue",
                            "size": 10,
                            "angle": winddir,
                        },
                        name="Wind speed",
                        showlegend=True,
                        hovertemplate=f"Wind speed: %{{y}}MPH<br>Wind Direction: {winddir:.0f}<br>%{{x|%I:%M %p}}<extra></extra>",
                    )
                )

        ## add traces for windspeed, and predicted windspeed
        wind_fig.add_traces(wind_scatters)

        return wind_fig
    except Exception as ex:
        func_name = "make_wind_fig"
        logging.error(f"Exception in {func_name}: \n{ex}")


def make_rain_fig(
    weather_df: pl.DataFrame,
    plot_start_date: datetime.datetime = None,
    plot_end_date: datetime.datetime = None,
) -> go.Figure:
    try:
        plot_df = filter_weather_df_by_date(weather_df, plot_start_date, plot_end_date)
        plot_start_date = plot_df["DATE"].min()
        plot_end_date = plot_df["DATE"].max()

        min_rain = 0
        if weather_df["RAIN_DAILIY"].max() is not None:
            max_rain = weather_df["RAIN_DAILIY"].max() + RAIN_PLOT_PADDING
        else:
            max_rain = RAIN_PLOT_PADDING

        ## my attempt to use a second y axis for the PRECIP_PROB 0:100 caused issues with
        ##   getting the hover template and background to show properly
        ##   so I'm just going to scale the PRECIP_PROB to match the max rain of the figure
        ##   then instead of labeling with PRECIP_PROB_SCALED i can pass PRECIP_PROB to
        ##   custom data and use that to make the popup
        plot_df = plot_df.with_columns(
            (pl.col("PRECIP_PROB") / 100 * max_rain).alias("PRECIP_PROB_SCALED")
        )

        ## create the figure
        rain_fig = go.Figure(
            layout=go.Layout(
                height=PLOT_FIG_HEIGHT,
                width=PLOT_FIG_WIDTH,
                paper_bgcolor=PAPER_BG_COLOR,
                showlegend=False,
                yaxis=go.layout.YAxis(
                    range=(min_rain, max_rain), title="Inches/Probability of Rain"
                ),
                xaxis=go.layout.XAxis(
                    range=(plot_start_date, plot_end_date),
                    tickformat="%-I:%M %p",
                    rangeslider={"visible": False},
                ),
                margin=go.layout.Margin(autoexpand=True, l=25, r=25, t=20, b=20),
            )
        )
        ## add traces for hourly rain and precip prob
        rain_actual_df = plot_df.filter(pl.col("PREDICTION").is_null())
        rain_prediction_df = plot_df.filter(pl.col("PREDICTION") == True)  # noqa: E712

        rain_fig.add_traces(
            [
                go.Scattergl(
                    x=rain_actual_df["DATE"],
                    y=rain_actual_df["RAIN_DAILIY"],
                    line=dict(color="rgb(0,0,255)"),
                    mode="lines",
                    showlegend=True,
                    hovertemplate='Daily Rain: %{y}"<br>%{x|%I:%M %p}<extra></extra>',
                    yaxis="y",
                ),
                go.Scattergl(
                    x=rain_prediction_df["DATE"],
                    y=rain_prediction_df["PRECIP_PROB_SCALED"],
                    customdata=rain_prediction_df["PRECIP_PROB"],
                    line=dict(color="rgb(0,0,255)"),
                    mode="lines",
                    fill="tozeroy",
                    showlegend=True,
                    hovertemplate="Probability of Rain: %{customdata} <br>%{x|%I:%M %p}<extra></extra>",
                ),
            ]
        )

        rain_fig = add_background_shading_to_figure(
            rain_fig, plot_start_date, plot_end_date
        )
        return rain_fig
    except Exception as ex:
        func_name = "make_rain_fig"
        logging.error(f"Exception in {func_name}: \n{ex}")


def make_winddir_polar_fig(current_windspeed, current_winddir):
    try:
        wind_df = pl.DataFrame(
            [[current_windspeed, current_winddir]],
            schema=["current_windspeed", "current_winddir"],
        )
        max_wind = max([10, current_windspeed])
        winddir_polar_fig = px.scatter_polar(
            wind_df,
            r="current_windspeed",
            theta="current_winddir",
            range_r=[0, max_wind],
            template="darkly",
            title=f"Current Wind {current_windspeed} MPH",
            width=200,
            height=200,
        )
        winddir_polar_fig.update_traces(marker={"size": 20, "color": "dodgerblue"})
        winddir_polar_fig.update_layout(title_x=0.5)
        winddir_polar_fig.layout.paper_bgcolor = "#222222"

        winddir_polar_fig.update_layout(
            margin=dict(l=1, r=1, t=45, b=25),
        )

        return winddir_polar_fig
    except Exception as ex:
        func_name = "make_winddir_polar_fig"
        logging.error(f"Exception in {func_name}: \n{ex}")


def weather_dicts_to_df(df_dicts: List[dict]) -> pl.DataFrame:
    try:
        return pl.DataFrame(df_dicts, WEATHER_DF_SCHEMA).with_columns(
            ## transform DATE string to datetime
            pl.col("DATE").str.strptime(
                pl.Datetime,
                format="%Y-%m-%dT%H:%M:%S",
                strict=False,
            )
        )
    except Exception as ex:
        func_name = "weather_dicts_to_df"
        logging.error(f"Exception in {func_name}: \n{ex}")


orig_weather_df = make_weather_df()
orig_weather_fig = make_temp_fig(orig_weather_df)
orig_wind_fig = make_wind_fig(orig_weather_df)
orig_rain_fig = make_rain_fig(orig_weather_df)

dark_theme = {
    "dark": True,
    "detail": "#007439",
    "primary": "#00EA64",
    "secondary": "#6E6E6E",
    "backgroundColor": "#292626",
}


server = flask.Flask(__name__)  # define flask app.server
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    server=server,
    update_title=None,
    suppress_callback_exceptions=True,
)  # call flask server
app.title = "Weather Dashboard"
current_temp = 0


def serve_layout():
    return html.Div(
        children=[
            dcc.Interval(
                id="interval-component",
                interval=60 * 1000 * 5,  # 5 min in milliseconds
                n_intervals=0,
            ),
            dcc.Interval(
                id="interval-component-small",
                interval=15 * 1000,  # 1/4 min in milliseconds
                n_intervals=0,
            ),
            dcc.Interval(
                id="interval-component-second",
                interval=1000,  # 1 sec in milliseconds
                n_intervals=0,
            ),
            dcc.Store(id="weather_df_store"),
            ## date/time, radar
            dbc.Row(
                [
                    dbc.Col(
                        [
                            daq.LEDDisplay(
                                id="time",
                                label="Time",
                                value="",
                                size=48,
                                color=dark_theme["primary"],
                                backgroundColor=dark_theme["backgroundColor"],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="month",
                                                label="Month",
                                                value="",
                                                size=24,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="day",
                                                label="Day",
                                                value="",
                                                size=24,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="year",
                                                label="Year",
                                                value="",
                                                size=24,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            html.Img(
                                id="radarimg",
                                src=app.get_asset_url("radar.gif"),
                                style={"height": "280px", "width": "280px"},
                            )
                        ]
                    ),
                ]
            ),
            ## Humidity, pressure, UV, radiation
            ## plots
            dbc.Row(
                [
                    dbc.Col(
                        [
                            daq.Thermometer(
                                id="current_temp_thermometer",
                                color="black",
                                showCurrentValue=True,
                                label="",
                                value=0,
                                height=125,
                                width=12,
                                max=110,
                                min=-10,
                                style={
                                    "display": "flex",
                                    "flex-direction": "row",
                                    "margin-left": 25,
                                    "margin-right": -5,
                                },
                                scale={
                                    "start": -10,
                                    "interval": 20,
                                    "labelInterval": 20,
                                },
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dash.html.Label(
                                "Observed",
                                style={
                                    "display": "flex",
                                    "flex-direction": "row",
                                    "margin-left": -15,  # Or whatever number suits your needs
                                    "margin-right": -15,
                                },
                            ),
                            dash.html.Label(
                                "High",
                                style={
                                    "display": "flex",
                                    "flex-direction": "row",
                                    "margin-left": -15,  # Or whatever number suits your needs
                                    "margin-right": -15,
                                },
                            ),
                            daq.LEDDisplay(
                                id="todays_high_temp_display",
                                label="",
                                value="",
                                size=10,
                                color=dark_theme["primary"],
                                backgroundColor=dark_theme["backgroundColor"],
                                style={
                                    "display": "flex",
                                    "flex-direction": "row",
                                    "margin-left": -15,  # Or whatever number suits your needs
                                    "margin-right": -15,
                                },
                            ),
                            daq.LEDDisplay(
                                id="todays_low_temp_display",
                                label="Low",
                                value="",
                                size=10,
                                color=dark_theme["primary"],
                                backgroundColor=dark_theme["backgroundColor"],
                                style={
                                    "display": "flex",
                                    "flex-direction": "row",
                                    "margin-left": -15,  # Or whatever number suits your needs
                                    "margin-right": -15,
                                },
                            ),
                        ],
                        align="start",
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="temp_graph",
                                figure=orig_weather_fig,
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="winddir_polar_graph",
                                figure=make_winddir_polar_fig(0, 0),
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="windspeed_graph",
                                figure=orig_wind_fig,
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="current_humidity_display",
                                                label="Humidity",
                                                value="00",
                                                size=ROW_TWO_SIZE,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flex-direction": "row",
                                                    "margin-left": 15,  # Or whatever number suits your needs
                                                    "margin-right": -15,
                                                },
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="current_pressure_display",
                                                label="Pressure",
                                                value="00.00",
                                                size=ROW_TWO_SIZE,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flex-direction": "row",
                                                    "margin-left": -15,  # Or whatever number suits your needs
                                                    "margin-right": -15,
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="current_UV_display",
                                                label="UV",
                                                value="000",
                                                size=ROW_TWO_SIZE,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flex-direction": "row",
                                                    "margin-left": 15,  # Or whatever number suits your needs
                                                    "margin-right": -15,
                                                },
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="current_radiation_display",
                                                label="Radiation",
                                                value="000",
                                                size=ROW_TWO_SIZE,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flex-direction": "row",
                                                    "margin-left": -15,  # Or whatever number suits your needs
                                                    "margin-right": -15,
                                                },
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            daq.LEDDisplay(
                                                id="current_rain_display",
                                                label="Daily Rain",
                                                value="00.00",
                                                size=ROW_TWO_SIZE,
                                                color=dark_theme["primary"],
                                                backgroundColor=dark_theme[
                                                    "backgroundColor"
                                                ],
                                                style={
                                                    "display": "flex",
                                                    "flex-direction": "row",
                                                    "margin-left": -15,  # Or whatever number suits your needs
                                                    "margin-right": -15,
                                                },
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="rain_graph",
                                figure=orig_rain_fig,
                                config={"displayModeBar": False},
                            ),
                        ]
                    ),
                ]
            ),
        ]
    )


app.layout = serve_layout


@app.callback(
    Output("time", "value"),
    Output("month", "value"),
    Output("day", "value"),
    Output("year", "value"),
    Input("interval-component-second", "n_intervals"),
)
def update_datetime(*args):
    time = f"{datetime.datetime.now().strftime('%I:%M')}"
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    year = datetime.datetime.now().year
    return time, month, day, year


@app.callback(
    Output("current_temp_thermometer", "value"),
    Output("current_temp_thermometer", "color"),
    Output("winddir_polar_graph", "figure"),
    Output("current_humidity_display", "value"),
    Output("current_pressure_display", "value"),
    Output("current_rain_display", "value"),
    Output("current_UV_display", "value"),
    Output("current_radiation_display", "value"),
    Input("interval-component-small", "n_intervals"),
)
def update_gauges_and_displays(*args):
    (
        current_temp,
        current_windspeed,
        current_winddir,
        humidity,
        gust,
        pressure,
        rain,
        uv,
        radiation,
    ) = query_most_recent_data()
    if current_temp < 0:
        temp_color = "purple"
    elif current_temp < 32 and current_temp >= 0:
        temp_color = "dodgerblue"
    elif current_temp < 50 and current_temp >= 32:
        temp_color = "green"
    elif current_temp < 70 and current_temp >= 50:
        temp_color = "yellow"
    elif current_temp < 90 and current_temp >= 70:
        temp_color = "orange"
    else:
        temp_color = "red"
    winddir_polar_fig = make_winddir_polar_fig(current_windspeed, current_winddir)

    current_windspeed = f"{current_windspeed:04.1f}"
    current_winddir = f"{current_winddir:03.0f}"
    return (
        current_temp,
        temp_color,
        winddir_polar_fig,
        humidity,
        pressure,
        rain,
        uv,
        radiation,
    )  # current_windspeed, current_winddir,


@app.callback(
    Output("weather_df_store", "data"), Input("interval-component", "n_intervals")
)
def update_weather_df_store(*args):
    weather_df = make_weather_df()
    return weather_df.to_dicts()


@app.callback(Output("radarimg", "src"), Input("interval-component", "n_intervals"))
def update_radar_img(*args):
    with open(radar_img_download_path, "rb") as image_file:
        img_data = base64.b64encode(image_file.read())
        img_data = img_data.decode()
        img_data = "{}{}".format("data:image/gif;base64, ", img_data)
    return img_data


@app.callback(
    Output("temp_graph", "figure"),
    Input("weather_df_store", "data"),
    prevent_initial_call=True,
)
def update_temp_fig(df_dicts):
    weather_df = weather_dicts_to_df(df_dicts)
    temp_fig = make_temp_fig(weather_df)
    return temp_fig


@app.callback(
    Output("todays_high_temp_display", "value"),
    Output("todays_low_temp_display", "value"),
    # Output('todays_high_windspeed_display', 'value'),
    Input("interval-component-small", "n_intervals"),
)
def update_high_low(*args):
    todays_high_temp, todays_low_temp, todays_high_windspeed = query_high_low()
    return todays_high_temp, todays_low_temp  # , todays_high_windspeed


@app.callback(
    Output("windspeed_graph", "figure"),
    Output("rain_graph", "figure"),
    Input("temp_graph", "relayoutData"),
    Input("weather_df_store", "data"),
)
def filter_other_plots(relayoutdata, df_dicts):
    try:
        weather_df = weather_dicts_to_df(df_dicts)
        if relayoutdata is not None and "xaxis.range[0]" in relayoutdata.keys():
            try:
                ## not sure what's up with the formatting of the range,
                ## sometimes it has a space and decimal seconds
                x_min = datetime.datetime.strptime(
                    relayoutdata["xaxis.range[0]"], "%Y-%m-%d %H:%M:%S.%f"
                )
                x_max = datetime.datetime.strptime(
                    relayoutdata["xaxis.range[1]"], "%Y-%m-%d %H:%M:%S.%f"
                )
                return make_wind_fig(weather_df, x_min, x_max), make_rain_fig(
                    weather_df, x_min, x_max
                )
            except Exception:
                try:
                    ## and others it has a T and decimals...
                    x_min = datetime.datetime.strptime(
                        relayoutdata["xaxis.range[0]"], "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    x_max = datetime.datetime.strptime(
                        relayoutdata["xaxis.range[1]"], "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    return make_wind_fig(weather_df, x_min, x_max), make_rain_fig(
                        weather_df, x_min, x_max
                    )
                except Exception:
                    try:
                        ## and others it has a T and no decimals...
                        x_min = datetime.datetime.strptime(
                            relayoutdata["xaxis.range[0]"], "%Y-%m-%dT%H:%M:%S"
                        )
                        x_max = datetime.datetime.strptime(
                            relayoutdata["xaxis.range[1]"], "%Y-%m-%dT%H:%M:%S"
                        )
                        return make_wind_fig(weather_df, x_min, x_max), make_rain_fig(
                            weather_df, x_min, x_max
                        )
                    except Exception:
                        ## and others it has a space and no decimals...
                        x_min = datetime.datetime.strptime(
                            relayoutdata["xaxis.range[0]"], "%Y-%m-%d %H:%M:%S"
                        )
                        x_max = datetime.datetime.strptime(
                            relayoutdata["xaxis.range[1]"], "%Y-%m-%d %H:%M:%S"
                        )
                        return make_wind_fig(weather_df, x_min, x_max), make_rain_fig(
                            weather_df, x_min, x_max
                        )
        else:
            return make_wind_fig(weather_df), make_rain_fig(weather_df)
    except Exception as ex:
        print(f"Exception in filter_other_plots {ex}")
        return make_wind_fig(weather_df), make_rain_fig(weather_df)


if __name__ == "__main__":
    app.run(debug=False)

    ##for windows waitress
    ##serve(app.server, host="0.0.0.0", port=8001, threads=10)
