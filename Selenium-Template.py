import undetected_chromedriver as uc
from json import loads
from math import ceil
import os

from selenium import webdriver
import requests

# import chromedriver_autoinstaller
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(800, 800))  
# display.start()
# chromedriver_autoinstaller.install()  # Check if the current version of chromedriver exists
                                      # and if it doesn't exist, download it automatically,
                                      # then add chromedriver to path

WOOCOMERCE_KEY=os.getenv("WOOCOMERCE_KEY")
WOOCOMERCE_SECRET=os.getenv("WOOCOMERCE_SECRET")
res = requests.get("https://zardaan.com/wp-json/torob/v1/listProds")

chrome_options = webdriver.ChromeOptions()
# # Add your options as needed
options = [
# Define window size here
    "--window-size=1200,1200",
    "--ignore-certificate-errors"

    #"--headless",
    #"--window-size=1920,1200",
    #"--ignore-certificate-errors",
    #"--disable-extensions",
    # These flags BELOW are recommended for stability when running Chrome in headless or containerized environments (such as GitHub Actions).
    "--ignore-certificate-errors",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    '--remote-debugging-port=9222',
    "--allow-insecure-localhost"
]
for option in options:
    chrome_options.add_argument(option)
driver = uc.Chrome(headless=True)
data =res.json()
f = open("result.txt","w")
for prod in data["response"]:
    key = prod["meta_value"]

    driver.get(f"https://api.torob.com/v4/base-product/sellers/?prk={key}")
    elm = driver.find_element(uc.By.XPATH,value="/html/body/pre")

    data = loads(elm.text)
    minIdx = -1
    count = 0
    tehrans = 0 
    noCapitals = 0
    accPrice = 0

    for i,res in enumerate(data["results"]):
        if res["price"]!=0:
            count+=1
            accPrice +=res["price"]
            if res["shop_name2"]=="تهران":
                tehrans+=1
            else:
                noCapitals+=1
    rounding = 100000
    avg = accPrice/count
    # all non capital
    if tehrans==0:
        expected=avg*1.11
    # all capital
    elif  noCapitals ==0:
        expected=avg*1.05
    else:
        if count>5:
            expected=avg*1.07
        else:
            expected=avg*1.09
    #roundings
    expectedRound = ceil(expected/rounding)*rounding
    body = {
            "id":prod["post_id"],
            "price":expectedRound,
    } 
    pageResponse = requests.post(f"https://zardaan.com/wp-json/torob/v1/UPDATE/",data=body)
    f.write(pageResponse.text)
f.close()