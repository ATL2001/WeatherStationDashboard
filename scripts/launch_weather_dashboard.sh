#!/bin/bash

cd /home/aaron/weather_assets/dashboard
source /home/aaron/weather_env/bin/activate
gunicorn weather_dash_polars:server -b 0.0.0.0:8050
