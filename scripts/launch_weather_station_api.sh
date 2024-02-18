#!/bin/bash

cd /home/aaron/weather_assets/api
source /home/aaron/weather_env/bin/activate
uvicorn main:app --host 0.0.0.0
