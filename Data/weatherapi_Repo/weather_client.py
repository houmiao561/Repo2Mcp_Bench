
import requests


def get_response(url, **kargs):
    response = requests.get(url)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


BASE_URL = 'http://api.weatherapi.com/v1'


def current_weather(q, api_key='', lang='en'):
    url = BASE_URL+f'/current.json?key={api_key}&q={q}&lang={lang}'
    return get_response(url)


def forecast(q, days=1, api_key='', lang='en', dt=None, unixdt=None, alerts=False, aqi=False, tp=False, hour=None):
    days = int(days)
    if days < 1:
        days = 1
    elif days > 14:
        days = 14
    url = BASE_URL + \
        f'/forecast.json?key={api_key}&q={q}&lang={lang}&days={days}'
    if dt != None:
        url += f"&dt={dt}"
    elif unixdt != None:
        url += f"&unixdt={unixdt}"
    if alerts == True:
        url += f"&alerts=yes"
    else:
        url += f"&alerts=no"
    if aqi == True:
        url += f"&aqi=yes"
    else:
        url += f"&aqi=no"
    if tp == True:
        url += f"&tp=15"
    if hour != None:
        url += f"&hour={hour}"
    return get_response(url)


def history(q, dt, api_key='', lang='en', unixdt=None, end_dt=None, unixend_dt=None, tp=False, hour=None):
    url = BASE_URL+f'/history.json?key={api_key}&q={q}&lang={lang}'
    url += f"&dt={dt}"
    if unixdt != None:
        url += f"&unixdt={unixdt}"
    if end_dt != None:
        url += f"&end_dt={end_dt}"
    elif unixend_dt != None:
        url += f"&unixend_dt={unixend_dt}"
    if hour != None:
        url += f"&hour={hour}"
    if tp == True:
        url += f"&tp=15"
    return get_response(url)


def marine(q, api_key='', lang='en', tides=False):
    url = BASE_URL+f'/marine.json?key={api_key}&q={q}&lang={lang}'
    if tides == True:
        url += f"&tides=yes"
    else:
        url += f"&tides=no"
    return get_response(url)


def future(q, dt, api_key='', lang='en'):
    url = BASE_URL+f'/future.json?key={api_key}&q={q}&lang={lang}&dt={dt}'
    print(url)
    return get_response(url)


def search(q, api_key='', lang='en'):
    url = BASE_URL+f'/search.json?key={api_key}&q={q}&lang={lang}'
    print(url)
    return get_response(url)


def ip_lookup(q, api_key='', lang='en'):
    url = BASE_URL+f'/ip.json?key={api_key}&q={q}&lang={lang}'
    print(url)
    return get_response(url)


def astronomy_info(q, api_key='', lang='en'):
    url = BASE_URL+f'/astronomy.json?key={api_key}&q={q}&lang={lang}'
    return get_response(url)


def time_zone(q, api_key='', lang='en'):
    url = BASE_URL+f'/timezone.json?key={api_key}&q={q}&lang={lang}'
    return get_response(url)


def sports(q, api_key='', lang='en'):
    url = BASE_URL+f'/sports.json?key={api_key}&q={q}&lang={lang}'
    return get_response(url)
