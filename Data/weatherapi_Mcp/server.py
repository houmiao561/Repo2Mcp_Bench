"""
WeatherAPI MCP Server
Exposes weatherapi.com endpoints as MCP tools via FastMCP.
"""
import os
import requests
from fastmcp import FastMCP

mcp = FastMCP("WeatherAPI Service")

API_KEY = os.environ.get("WEATHER_API_KEY", "")
BASE_URL = "http://api.weatherapi.com/v1"


def _get(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError:
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except ValueError:
        return {"error": "JSON parse failed", "raw": resp.text[:500]}


@mcp.tool
def current_weather(q: str, lang: str = "en") -> dict:
    """Get up-to-date current weather information for a given location.

    Args:
        q: Query — city name, lat/lon, US/UK/Canada postal code, IP address,
           or 'auto:ip' for auto-detect. Examples: 'Paris', '48.8567,2.3508'.
        lang: Language code for the 'condition:text' field. Default 'en'.
    """
    return _get(f"{BASE_URL}/current.json?key={API_KEY}&q={q}&lang={lang}")


@mcp.tool
def forecast(
    q: str,
    days: int = 1,
    lang: str = "en",
    dt: str | None = None,
    unixdt: str | None = None,
    alerts: bool = False,
    aqi: bool = False,
    tp: bool = False,
    hour: int | None = None,
) -> dict:
    """Get weather forecast for up to 14 days including weather alerts.

    Args:
        q: Location query (city name, coordinates, postal code, IP, etc.).
        days: Number of forecast days (1-14).
        lang: Language code. Default 'en'.
        dt: Restrict output to a specific date in yyyy-MM-dd format.
        unixdt: Same as dt but as Unix timestamp. Use either dt or unixdt, not both.
        alerts: Enable weather alerts in output.
        aqi: Enable Air Quality data in output.
        tp: Get 15-minute interval data (Enterprise only).
        hour: Restrict output to a specific hour (0-23, 24h format).
    """
    days = max(1, min(14, int(days)))
    url = f"{BASE_URL}/forecast.json?key={API_KEY}&q={q}&lang={lang}&days={days}"
    if dt is not None:
        url += f"&dt={dt}"
    elif unixdt is not None:
        url += f"&unixdt={unixdt}"
    url += f"&alerts={'yes' if alerts else 'no'}"
    url += f"&aqi={'yes' if aqi else 'no'}"
    if tp:
        url += "&tp=15"
    if hour is not None:
        url += f"&hour={hour}"
    return _get(url)


@mcp.tool
def history(
    q: str,
    dt: str,
    lang: str = "en",
    unixdt: str | None = None,
    end_dt: str | None = None,
    unixend_dt: str | None = None,
    tp: bool = False,
    hour: int | None = None,
) -> dict:
    """Get historical weather data for a date on or after 2010-01-01.

    Args:
        q: Location query.
        dt: Date in yyyy-MM-dd format (required).
        lang: Language code. Default 'en'.
        unixdt: Same as dt but as Unix timestamp. Use either dt or unixdt, not both.
        end_dt: End date in yyyy-MM-dd format. Must be > dt, max 30 days apart. Pro plan+.
        unixend_dt: Same as end_dt but as Unix timestamp.
        tp: Get 15-minute interval data (Enterprise only).
        hour: Restrict output to a specific hour (0-23).
    """
    url = f"{BASE_URL}/history.json?key={API_KEY}&q={q}&lang={lang}&dt={dt}"
    if unixdt is not None:
        url += f"&unixdt={unixdt}"
    if end_dt is not None:
        url += f"&end_dt={end_dt}"
    elif unixend_dt is not None:
        url += f"&unixend_dt={unixend_dt}"
    if hour is not None:
        url += f"&hour={hour}"
    if tp:
        url += "&tp=15"
    return _get(url)


@mcp.tool
def marine(q: str, lang: str = "en", tides: bool = False) -> dict:
    """Get marine and sailing weather forecast and optional tide data.

    Args:
        q: Location query (preferably lat/lon for ocean points).
        lang: Language code. Default 'en'.
        tides: Enable tide data (Pro plan and above only).
    """
    url = f"{BASE_URL}/marine.json?key={API_KEY}&q={q}&lang={lang}"
    url += f"&tides={'yes' if tides else 'no'}"
    return _get(url)


@mcp.tool
def future(q: str, dt: str, lang: str = "en") -> dict:
    """Get future weather for a date between 14 and 300 days from today.

    Args:
        q: Location query.
        dt: Future date in yyyy-MM-dd format (14-300 days from today).
        lang: Language code. Default 'en'.
    """
    return _get(f"{BASE_URL}/future.json?key={API_KEY}&q={q}&lang={lang}&dt={dt}")


@mcp.tool
def search(q: str, lang: str = "en") -> dict:
    """Search for matching cities and towns. Returns an array of locations.

    Args:
        q: City or location name to search for.
        lang: Language code. Default 'en'.
    """
    return _get(f"{BASE_URL}/search.json?key={API_KEY}&q={q}&lang={lang}")


@mcp.tool
def ip_lookup(q: str, lang: str = "en") -> dict:
    """Get location and other info for an IP address.

    Args:
        q: IP address (IPv4 or IPv6).
        lang: Language code. Default 'en'.
    """
    return _get(f"{BASE_URL}/ip.json?key={API_KEY}&q={q}&lang={lang}")


@mcp.tool
def astronomy_info(q: str, lang: str = "en") -> dict:
    """Get sunrise, sunset, moonrise, moonset, moon phase and illumination.

    Args:
        q: Location query.
        lang: Language code. Default 'en'.
    """
    return _get(f"{BASE_URL}/astronomy.json?key={API_KEY}&q={q}&lang={lang}")


@mcp.tool
def time_zone(q: str, lang: str = "en") -> dict:
    """Get time zone and local time information for a location.

    Args:
        q: Location query.
        lang: Language code. Default 'en'.
    """
    return _get(f"{BASE_URL}/timezone.json?key={API_KEY}&q={q}&lang={lang}")


@mcp.tool
def sports(q: str, lang: str = "en") -> dict:
    """Get upcoming sports events (football, cricket, golf) for a location.

    Args:
        q: Location query.
        lang: Language code. Default 'en'.
    """
    return _get(f"{BASE_URL}/sports.json?key={API_KEY}&q={q}&lang={lang}")


if __name__ == "__main__":
    import argparse, asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    asyncio.run(mcp.run_sse_async(host=args.host, port=args.port))
