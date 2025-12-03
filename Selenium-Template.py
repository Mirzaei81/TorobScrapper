import undetected_chromedriver as uc
from json import loads
import os
import pandas

from selenium import webdriver
import requests
import json

def get_sku_name(id):
    url = f"https://zardaan.com/wp-json/wc/v3/products/{id}"
    payload = json.dumps({
    "searchParameters": {
        "input": "00138432",
        "type": "QUERY"
    },
    "components": [
        {
        "component": "PRIMARY_AREA"
        }
    ]
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Basic Y2tfYTdjNGVlM2U5NTc1MDI4MWQ5MTg1MmRlOTJkMjc1NWNkMDUyZGUyMjpjc18yNWU4NDQ4YzZkMWE1YzdkYTlhMGFlMDE0Y2M4ZWQ2YzViMGU2MWE5',
    'Cookie': 'pxcelPage_c01002=1'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return response.json()

import chromedriver_autoinstaller
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(800, 800))  
# display.start()
chromedriver_autoinstaller.install()  # Check if the current version of chromedriver exists
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
    "--ignore-certificate-errors",
    "--disable-gpu",
    "--no-sandbox",
    "--allow-insecure-localhost"
]
for option in options:
    chrome_options.add_argument(option)
driver = uc.Chrome(headless=True)
data =res.json()
productData  = {"prices":[],"names":[],"sku":[],"ids":[],"urls":[],"locs":[],"shops":[]}

for prod in data["response"]:
    key = prod["meta_value"]
    driver.get(f"https://api.torob.com/v4/base-product/sellers/?prk={key}")
    zardanProd = get_sku_name(prod["post_id"])
    if zardanProd["brands"] and zardanProd["brands"][0]["id"]==7328:
        elm = driver.find_element(uc.By.XPATH,value="/html/body/pre")

        data = loads(elm.text)
        for i,res in enumerate(data["results"]):
            if  not res["availability"]:
                break
            if res["price"]!=0:
                productData["ids"].append(zardanProd["id"])
                productData["names"].append(zardanProd["slug"])
                productData["sku"].append(zardanProd["sku"])
                productData["prices"].append(res["price"])
                productData["shops"].append(res["shop_name"])
                productData["urls"].append(zardanProd["permalink"])
                if res["shop_name2"]=="تهران":
                    productData["locs"].append("T")
                else:
                    productData["locs"].append("nT")
df = pandas.DataFrame(productData)
df.to_csv("product_data.csv",encoding="utf-8-sig")
