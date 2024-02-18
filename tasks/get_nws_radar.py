import logging

import requests

logname = r"/home/aaron/weather_assets/logs/radar.log"
radar_img_src = r"https://radar.weather.gov/ridge/standard/KLSX_loop.gif"
radar_img_download_path = r"/home/aaron/weather_assets/dashboard/assets/radar.gif"

logging.basicConfig(
    filename=logname,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

try:
    img_data = requests.get(radar_img_src).content
    with open(radar_img_download_path, "wb") as handler:
        handler.write(img_data)
    logging.info("Updated radar gif.")
except Exception as ex:
    logging.error(f"Something died getting radar from nws. {ex}")
