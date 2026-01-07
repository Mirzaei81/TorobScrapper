from json import loads
import os
import pandas
import requests
import json
import re
from collections import defaultdict

from unidecode import unidecode
from botasaurus.browser import browser,Driver

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


@browser(    
        block_images_and_css=True,
        wait_for_complete_page_load=False,
)
def scrape_heading_task(driver: Driver, data):
    driver.long_random_sleep()
    driver.enable_human_mode()
    driver.google_get(data,bypass_cloudflare=True)
    driver.long_random_sleep()
    driver.disable_human_mode()
    return driver.get_text("body")

WOOCOMERCE_KEY=os.getenv("WOOCOMERCE_KEY")
WOOCOMERCE_SECRET=os.getenv("WOOCOMERCE_SECRET")
session = requests.Session()
res = session.get("https://zardaan.com/wp-json/torob/v1/listProds")

agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"


data =res.json()
productData  =defaultdict(list)
pat = "\(([\u0660-\u0669\u06F0-\u06F9]+) (سال|ماه).*\)"
# with SB(incognito=True,agent=agent,headless=False,headed=True,xvfb=use_xvfb,uc=True, test=True,locale="fa") as sb:
#     sb.execute_script(
#         "try{Object.defineProperty(navigator, 'webdriver', {get: () => undefined});}catch(e){}"
#     )
for prod in data["response"]:
    key = prod["meta_value"]
    try:
        body = scrape_heading_task(data=f"https://api.torob.com/v4/base-product/sellers/?prk={key}")
        zardanProd = get_sku_name(prod["post_id"])

        data = loads(body)
        for i,res in enumerate(data["results"]):
            if  not res["availability"] or res["is_price_unreliable"]:
                break
            if res["price"]!=0:
                continue
            try:
                title  = res["score_info"]["complaints_info"]["title"]
                if title:
                    if title=="فروشگاه جدید":
                        continue
                    g = re.search(pat,title)
                    if g and  int(unidecode(g.group(1)))<6 and g.group(2)=="ماه":
                        continue
            except Exception as e:
                print(e)
                continue
            productData["ids"].append(zardanProd["id"])
            productData["names"].append(zardanProd["slug"])
            productData["sku"].append(zardanProd["sku"])
            productData["prices"].append(res["price"])
            productData["shops"].append(res["shop_name"])
            productData["urls"].append(zardanProd["permalink"])
            productData["parent_id"].append(prod["post_parent"])
            productData["unreliable"].append(res["is_price_unreliable"])
            productData["locs"].append(res["shop_name2"])
            if (zardanProd["brands"]):
                productData["brands"].append(zardanProd["brands"][0]["id"])
            else:
                productData["brands"].append(-1)
    except Exception as e:
        print(f"Erorr while fetching for {prod} ,{e}")
df = pandas.DataFrame(productData)
df.to_csv("product_data.csv",encoding="utf-8-sig")
