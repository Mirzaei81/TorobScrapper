import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import os 
import dotenv
dotenv.load_dotenv()

df = pd.read_csv("product_data.csv",encoding="utf-8")
CONFIG = {
    'Z_THRESHOLD': 3.5,           
    'NORMAL_THRESHOLD': 0.85,     
    'TEHRAN_NORMAL': 1.05,        
    'SHAHRESTAN_NORMAL': 1.08,    
    'TEHRAN_HIGH': 1.80,          
    'SHAHRESTAN_HIGH': 1.10,      
    'SHAHRESTAN_RATIO': 0.70,     
    'TEHRAN_RATIO': 0.60,         
    'ROUND_STEP': 50000,          
    'SHIPPING_RATE': 0.10,        
    'SHIPPING_CAP': 600000,       
    'TOLERANCE_ADD_MIN': 500000,  
    'TOLERANCE_ADD_MAX': 6000000, 
    'TOLERANCE_XIAOMI': 0.05,     
    'TOLERANCE_OTHER': 0.03,      
}


def weighted_winsor(values, weights, alpha):
    n = len(values)
    k = int(math.floor(alpha * n))

    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_weights = weights[order]

    if k > 0:
        low = sorted_vals[k]
        high = sorted_vals[-k-1]
        wins = np.clip(values, low, high)
    else:
        wins = values.copy()

    return wins


def weighted_trimmed_mean(values, weights, alpha):
    n = len(values)
    k = int(math.floor(alpha * n))

    order = np.argsort(values)
    vals = values[order]
    wts = weights[order]

    if k == 0:
        return float(np.sum(vals * wts) / np.sum(wts))

    vals_trim = vals[k:-k]
    wts_trim = wts[k:-k]

    return float(np.sum(vals_trim * wts_trim) / np.sum(wts_trim))

def combined_mean(values, weights, prev_mean=None,
                  alpha_winsor=0.05, alpha_trim=0.05, ema_factor=0.3):

    wins = weighted_winsor(values, weights, alpha_winsor)
    m1 = float(np.sum(wins * weights) / np.sum(weights))

    m2 = weighted_trimmed_mean(values, weights, alpha_trim)

    combined = 0.5 * m1 + 0.5 * m2

    if prev_mean is None:
        return combined
    else:
        # EMA نرم‌سازی
        return (1 - ema_factor) * prev_mean + ema_factor * combined

def optimize_prices(values, locations,
                    tehran_weight=1.4,
                    county_weight=1.0,
                    alpha_winsor=0.05,
                    alpha_trim=0.05,
                    replacement_count=5,
                    target_multiplier=1.15,
                    tehran_tolerance=0.07,
                    max_iters=20):

    v = np.array(values, dtype=float)
    locs = np.array(locations)
    if len(v)==0:
        return  0,
    if len(v)<=3:
        return (v.max()*1.1).item()

    weights = np.where(locs == 'T', tehran_weight, county_weight).astype(float)

    orig_min = v.min()
    target_min = target_multiplier * orig_min

    # میانگین تهران برای محدوده هدف
    if  len(v[locs == 'T'])!=0:
        tehran_mean = v[locs == 'T'].mean()
        low_t = tehran_mean * (1 - tehran_tolerance)
        high_t = tehran_mean * (1 + tehran_tolerance)
    else:
        tehran_mean = orig_min
        low_t = orig_min * (1-tehran_tolerance)
        high_t = orig_min * (1-tehran_tolerance)
    history = []

    current_mean = combined_mean(v, weights, prev_mean=None,
                                 alpha_winsor=alpha_winsor,
                                 alpha_trim=alpha_trim)

    history.append({"iter": 0, "mean": current_mean, "max": v.max()})

    for it in range(1, max_iters + 1):

        # شرط توقف: هر دو باید برقرار باشد
        cond1 = current_mean >= target_min
        cond2 = (current_mean >= low_t) and (current_mean <= high_t)

        if cond1 and cond2:
            break

        # پیدا کردن ۵ ماکزیمم
        max_indices = np.argsort(-v)[:replacement_count]

        # جایگزینی با میانگین فعلی
        for idx in max_indices:
            v[idx] = current_mean

        # محاسبه میانگین جدید
        prev = current_mean
        current_mean = combined_mean(v, weights, prev_mean=prev,
                                     alpha_winsor=alpha_winsor,
                                     alpha_trim=alpha_trim)

        history.append({
            "iter": it,
            "mean": current_mean,
            "max": v.max(),
            "replaced_indices": max_indices.tolist()
        })

    return current_mean if  len(v[locs == 'T'])!=0 else current_mean*1.2
def calculate_fair_values(data):
    # 1. حذف 10 درصد پایین و بالا داده‌ها
    data_sorted = np.sort(data)
    n = len(data_sorted)
    lower_bound = int(n * 0.1)
    upper_bound = int(n * 0.9)
    trimmed_data = data_sorted[lower_bound:upper_bound]
    
    # 2. محاسبه میانه
    median_value = np.median(trimmed_data)
    
    # 3. تعیین وزن‌ها: 1.2 به داده‌های کوچکتر (نیمه پایین)، 1 به داده‌های بزرگتر (نیمه بالا)
    mid_index = len(trimmed_data) // 2
    weights = np.ones(len(trimmed_data))
    weights[:mid_index] = 1.2
    
    # 4. محاسبه میانگین وزنی گاوس
    weighted_mean = np.ma.average(trimmed_data, weights=weights)
    
    # 5. محاسبه میانگین ساده میانه و میانگین وزنی
    combined_mean = (median_value + weighted_mean) / 2
    
    return float(combined_mean.item())


def weighted_trimmed_mean(values, weights, alpha):
    n = len(values)
    k = int(math.floor(alpha * n))

    order = np.argsort(values)
    vals = values[order]
    wts = weights[order]

    if k == 0:
        return float(np.sum(vals * wts) / np.sum(wts))

    vals_trim = vals[k:-k]
    wts_trim = wts[k:-k]

    return float(np.sum(vals_trim * wts_trim) / np.sum(wts_trim))

def combined_mean(values, weights, prev_mean=None,
                  alpha_winsor=0.05, alpha_trim=0.05, ema_factor=0.3):

    wins = weighted_winsor(values, weights, alpha_winsor)
    m1 = float(np.sum(wins * weights) / np.sum(weights))

    m2 = weighted_trimmed_mean(values, weights, alpha_trim)

    combined = 0.5 * m1 + 0.5 * m2

    if prev_mean is None:
        return combined
    else:
        # EMA نرم‌سازی
        return (1 - ema_factor) * prev_mean + ema_factor * combined

def optimize_prices(values, locations,
                    tehran_weight=1.4,
                    county_weight=1.0,
                    alpha_winsor=0.05,
                    alpha_trim=0.05,
                    replacement_count=5,
                    target_multiplier=1.15,
                    tehran_tolerance=0.07,
                    max_iters=20):

    v = np.array(values, dtype=float)
    locs = np.array(locations)
    if len(v)==0:
        return  0,
    if len(v)<=3:
        return (v.max()*1.1).item()

    weights = np.where(locs == 'T', tehran_weight, county_weight).astype(float)

    orig_min = v.min()
    target_min = target_multiplier * orig_min

    # میانگین تهران برای محدوده هدف
    if  len(v[locs == 'T'])!=0:
        tehran_mean = v[locs == 'T'].mean()
        low_t = tehran_mean * (1 - tehran_tolerance)
        high_t = tehran_mean * (1 + tehran_tolerance)
    else:
        tehran_mean = orig_min
        low_t = orig_min * (1-tehran_tolerance)
        high_t = orig_min * (1-tehran_tolerance)
    history = []

    current_mean = combined_mean(v, weights, prev_mean=None,
                                 alpha_winsor=alpha_winsor,
                                 alpha_trim=alpha_trim)

    history.append({"iter": 0, "mean": current_mean, "max": v.max()})

    for it in range(1, max_iters + 1):

        # شرط توقف: هر دو باید برقرار باشد
        cond1 = current_mean >= target_min
        cond2 = (current_mean >= low_t) and (current_mean <= high_t)

        if cond1 and cond2:
            break

        # پیدا کردن ۵ ماکزیمم
        max_indices = np.argsort(-v)[:replacement_count]

        # جایگزینی با میانگین فعلی
        for idx in max_indices:
            v[idx] = current_mean

        # محاسبه میانگین جدید
        prev = current_mean
        current_mean = combined_mean(v, weights, prev_mean=prev,
                                     alpha_winsor=alpha_winsor,
                                     alpha_trim=alpha_trim)

        history.append({
            "iter": it,
            "mean": current_mean,
            "max": v.max(),
            "replaced_indices": max_indices.tolist()
        })

    return current_mean if  len(v[locs == 'T'])!=0 else current_mean*1.2
def gaussian_fit(data):
    mu = np.mean(data)
    sigma = np.std(data)
    q4 = norm.ppf(q=0.4,loc=mu,scale=sigma)
    q5 = norm.ppf(q=0.5,loc=mu,scale=sigma)
    return ((q5-q4)//2).item()

def minimum_fair_price(prices):
    # حذف مقادیر صفر یا ناموجود
    filtered_prices = [p for p in prices if p and p > 0]

    # مرتب‌سازی داده‌ها
    prices_sorted = np.sort(filtered_prices)
    n = len(prices_sorted)
    if n < 3:
        return None  # یا پیغام مناسب
    
    # حذف 10 درصد ابتدا و انتهای لیست (پرت)
    lower = int(n * 0.1)
    upper = int(n * 0.9)
    trimmed = prices_sorted[lower:upper]

    # شرط 1: 10٪ بالاتر از حداقل قیمت تهران
    min_tehran = prices_sorted.min()
    cond_1 = min_tehran * 1.1

    # شرط 2: میانگین نرمال (گائوسی) پس از حذف پرت‌ها
    mean_gauss = np.mean(trimmed)
    cond_2 = mean_gauss

    # شرط 3: میانه داده‌های باقی‌مانده
    median_gauss = np.median(trimmed)
    cond_3 = median_gauss

    # شرط 4: کمتر از حداکثر قیمت تهران
    max_tehran = prices_sorted.max()
    cond_4 = max_tehran

    # شرط 5: 20٪ بالاتر از حداقل کل قیمت‌ها
    min_all = prices_sorted.min()
    cond_5 = min_all * 1.2

    # انتخاب بیشترین مقدار کف از شروط و کمتر از سقف بیشینه
    minimum_value = max(cond_1, cond_2, cond_3, cond_5)
    final_value = min(minimum_value, cond_4)

    # رند کردن به نزدیک‌ترین ضریب ۵۰,۰۰۰ تومان بالا
    rounded_value = int(np.ceil(final_value / 50000) * 50000)
    return rounded_value
def persian_to_english_digits(text):
    """تبدیل اعداد فارسی به انگلیسی"""
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    return text.translate(str.maketrans(persian_digits, english_digits))

def remove_outliers(price_list):
    """حذف outliers با روش IQR"""
    if not price_list:
        return []
    q1 = np.percentile(price_list, 25)
    q3 = np.percentile(price_list, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [p for p in price_list if lower <= p <= upper]

def calculate_competitive_price(prods):
    prods["estimated"] = np.where(prods["brands"]==7328,prods["prices"]*1.12,prods["prices"]*1.1)
    
    # ⿢ جداسازی تهران/شهرستان
    tehran_vals = prods.where(prods['locs']=="T")["estimated"]
    shahrestan_vals = prods.where(prods['locs']!="T")["estimated"]
    
    mu_T = np.mean(tehran_vals) if len(tehran_vals)!=0 else None
    mu_S = np.mean(shahrestan_vals) if len(shahrestan_vals)!=0 else None
    
    all_prices = tehran_vals + shahrestan_vals
    CI = (max(all_prices) - min(all_prices)) / np.mean(all_prices) if len(all_prices)!=0 else 1.0
    
    alpha_T = 0.03 if CI <= 0.03 else 0.05
    alpha_S = 0.05 if CI <= 0.03 else 0.08
    
    n_T = len(tehran_vals)
    n = len(prods)
    if n_T / n >= 0.7 and mu_T:
        price_final = 'T', mu_T * (1 + alpha_T)
    else:
        price_T = mu_T * (1 + alpha_T) if mu_T else float('inf')
        price_S = (mu_S + 500000) * (1 + alpha_S) if mu_S else float('inf')
        if price_S < price_T:
            s,price_final = price_S
        else:
            t,price_final =price_T
    
    return price_final[1]

# همه توابع کمکی (CONFIG, PROVINCES, detect_province, remove_outliers, get_tolerance_rate, 
# calculate_tolerance_addition, get_region_coefficients, select_strategy) همان قبلی‌ها
def get_tolerance_rate(brand_name: np.int64) -> float:
    """تلورانس تفکیکی"""
    if  brand_name==7328:
        return CONFIG['TOLERANCE_XIAOMI']
    return CONFIG['TOLERANCE_OTHER']

def calculate_tolerance_addition(base_value: float, rate: float) -> float:
    """تلورانس اضافه محدود 500k-6M"""
    suggested = base_value * rate
    return max(min(suggested, CONFIG['TOLERANCE_ADD_MAX']), CONFIG['TOLERANCE_ADD_MIN'])

def get_region_coefficients(shahrestan_ratio: float) -> tuple:
    """ضرایب بر اساس غلبه"""
    if shahrestan_ratio > CONFIG['SHAHRESTAN_RATIO']:
        return CONFIG['TEHRAN_HIGH'], CONFIG['SHAHRESTAN_HIGH']
    return CONFIG['TEHRAN_NORMAL'], CONFIG['SHAHRESTAN_NORMAL']

def select_strategy(tehran_ratio: float, shahrestan_ratio: float, 
                   tehran_price: float, shahrestan_price: float) -> tuple:
    """استراتژی 60%"""
    if tehran_ratio > CONFIG['TEHRAN_RATIO']:
        return max(tehran_price, shahrestan_price), "max (تهران>60%)"
    elif shahrestan_ratio > CONFIG['TEHRAN_RATIO']:
        return min(tehran_price, shahrestan_price), "min (شهرستان>60%)"
    return (tehran_price + shahrestan_price) / 2, "میانگین (تعادل)"



def calculate_competitive_price_tehran_floor(prod:pd.DataFrame):
    """الگوریتم نهایی - قیمت ≤ میانه + کف تهران حفظ شود"""
    
    if len(prod) < 3:
        return np.inf
    
    
    # 🔥 میانه کل (سقف نهایی)
    median_cap = prod['prices'].median()
    
    # 🔥 کف تهران (حداقل مطلق)
    tehran_prices = prod[prod['locs'] == 'تهران']['prices']
    tehran_floor = tehran_prices.min() if len(tehran_prices) > 0 else 0
    
    # 5. پاکسازی منطقه‌ای
    tehran_data = prod[prod['locs'] == 'تهران']
    other_data = prod[prod['locs'] != 'تهران']
    
    tehran_clean = tehran_data
    shahrestan_clean = other_data
    
    # 6. تلورانس (فیلیپس = 3%)
    brand = prod['brands'].iloc[0]
    tolerance_rate = get_tolerance_rate(brand)
    
    tehran_base = tehran_clean["prices"].min() if len(tehran_clean) > 0 else float('inf')
    shahrestan_base = shahrestan_clean["prices"].min() if len(shahrestan_clean) > 0 else float('inf')
    
    tehran_tolerance = calculate_tolerance_addition(tehran_base, tolerance_rate)
    shahrestan_tolerance = calculate_tolerance_addition(shahrestan_base, tolerance_rate)
    
    tehran_price = tehran_base + tehran_tolerance
    shahrestan_price = shahrestan_base + shahrestan_tolerance
    
    # 🔥 اعمال کف تهران
    tehran_price = max(tehran_price, tehran_floor)
    shahrestan_price = max(shahrestan_price, tehran_floor)
    
    # 7. ضرایب + استراتژی
    total_clean = len(tehran_clean) + len(shahrestan_clean)
    shahrestan_ratio = len(shahrestan_clean) / total_clean if total_clean > 0 else 0
    tehran_ratio = 1 - shahrestan_ratio
    
    coef_tehran, coef_shahrestan = get_region_coefficients(shahrestan_ratio)
    
    final_tehran = min(tehran_price * coef_tehran, median_cap)
    shipping = min(shahrestan_base * CONFIG['SHIPPING_RATE'], CONFIG['SHIPPING_CAP'])
    final_shahrestan = min(shahrestan_price * coef_shahrestan + shipping, median_cap)
    
    final_price, strategy = select_strategy(tehran_ratio, shahrestan_ratio, 
                                          final_tehran, final_shahrestan)
    
    # 🔥 محدودیت نهایی
    capped_price = max(min(final_price, median_cap), tehran_floor)
    rounded_price = round(capped_price / CONFIG['ROUND_STEP']) * CONFIG['ROUND_STEP']
    
    return int(rounded_price)


def applyOptimization(g):
    Q4 = g["prices"].quantile(1)
    Q1 = g["prices"].quantile(.25)
    mask = g["prices"].between(Q1, Q4, inclusive="both")
    filtered = g.loc[mask]
    if len(filtered)>0:
        return pd.Series({
                "winzor": optimize_prices(filtered["prices"], filtered["locs"]),
                "gaussian1": calculate_fair_values(filtered["prices"]),
                "gaussian2": minimum_fair_price(filtered["prices"]),
                "competetive_price":calculate_competitive_price(filtered),
                "competetive_price_2":calculate_competitive_price_tehran_floor(filtered),
                "sellerCount":len(filtered),
                "max":max(filtered["prices"]),
                "mean":filtered["prices"].mean(),
                "sku":max(filtered["sku"]),
                "parent_id":max(filtered["parent_id"])
            })




result:pd.DataFrame =  df.groupby("ids").apply(
		applyOptimization
)

rounding = 1e5
import requests

result = result.replace(np.nan,np.inf)
f = open("output.json","w")
ss ='{"result":['
def updateWeb():
    global ss
    for i,r in result.iterrows():
        rounded = round(min(r.competetive_price,r.winzor ,r.gaussian1,r.gaussian2,r.competetive_price_2,r["mean"])/rounding*1.17)*rounding
        body = {
            "regular_price":rounded,    
            "sale_price":rounded,
            "stock": 0 if r.sellerCount<5 else 10,  
            "backorders": "yes" if r.sellerCount<5 else "no", 
            "backorders_allowed": True if r.sellerCount<5 else False,    
            "stock_status": "onbackorder" if r.sellerCount<5 else "instock"
        } 
        headers = {
        'Authorization': 'Basic Y2tfYTdjNGVlM2U5NTc1MDI4MWQ5MTg1MmRlOTJkMjc1NWNkMDUyZGUyMjpjc18yNWU4NDQ4YzZkMWE1YzdkYTlhMGFlMDE0Y2M4ZWQ2YzViMGU2MWE5',
        }
        if r["parent_id"]!=0:
            pageResponse = requests.put(f"https://zardaan.com/wp-json/wc/v3/products/{r['parent_id']}/variations/{i}",data=body,headers=headers)
        else:
            pageResponse = requests.put(f"https://zardaan.com/wp-json/wc/v3/products/{i}",data=body,headers=headers)
        ss+=pageResponse.text+","
updateWeb()
ss = ss[:-1]+"]}"
f.write(ss)
