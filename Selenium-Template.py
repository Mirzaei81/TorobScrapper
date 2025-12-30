from json import loads
import os
import time
import pandas
import requests
import json
from seleniumbase import SB
from bs4 import BeautifulSoup

if os.name!="nt":
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(800, 600))
    display.start()



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

    response = session.get(url, headers=headers, data=payload)

    return response.json()


WOOCOMERCE_KEY=os.getenv("WOOCOMERCE_KEY")
WOOCOMERCE_SECRET=os.getenv("WOOCOMERCE_SECRET")
session = requests.Session()
res = session.get("https://zardaan.com/wp-json/torob/v1/listProds")

agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"


data =res.json()
productData  = {"prices":[],"names":[],"sku":[],"ids":[],"urls":[],"locs":[],"shops":[],"brands":[],"parent_id":[]}
with SB(incognito=True,agent=agent,headless=False,headed=False,uc=True, test=True) as sb:
    for prod in data["response"]:
        key = prod["meta_value"]
        try:
            sb.activate_cdp_mode(f"https://api.torob.com/v4/base-product/sellers/?prk={key}")
            time.sleep(7)
            sb.cdp.gui_click_element('#myWidget')
            time.sleep(3)
            zardanProd = get_sku_name(prod["post_id"])
            body = sb.get_html()
            soup = BeautifulSoup(body,"lxml")
            elm = soup.select_one("body > pre")

            data = loads(elm.text)
            for i,res in enumerate(data["results"]):
                if  not res["availability"] or res["is_price_unreliable"]:
                    break
                if res["price"]!=0:
                    productData["ids"].append(zardanProd["id"])
                    productData["names"].append(zardanProd["slug"])
                    productData["sku"].append(zardanProd["sku"])
                    productData["prices"].append(res["price"])
                    productData["shops"].append(res["shop_name"])
                    productData["urls"].append(zardanProd["permalink"])
                    productData["parent_id"].append(prod["post_parent"])
                    productData["locs"].append(res["shop_name2"])
                    if (zardanProd["brands"]):
                        productData["brands"].append(zardanProd["brands"][0]["id"])
                    else:
                        productData["brands"].append(-1)
        except Exception as e:
            print(f"Erorr while fetching for {prod} ,{e}")
df = pandas.DataFrame(productData)
df.to_csv("product_data.csv",encoding="utf-8-sig")
