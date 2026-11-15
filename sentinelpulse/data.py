from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd
import requests

# Explicitly fictional operational archetypes for the hackathon demonstration.
# These are NOT claims about the real industries, shifts, or worker populations
# in the named geographies. Replace with authorized site schedules before use.
DEMO_OPERATIONAL_PROFILES = {
    "Abu Dhabi Island": {"name": "Urban mixed-shift demo", "base": 0.48, "amplitude": 0.09, "exposure": 0.72, "start": 6, "end": 18},
    "Khalifa City": {"name": "Daytime logistics demo", "base": 0.62, "amplitude": 0.13, "exposure": 0.88, "start": 6, "end": 19},
    "Mussafah": {"name": "Extended industrial-shift demo", "base": 0.70, "amplitude": 0.07, "exposure": 0.96, "start": 5, "end": 21},
    "Al Ain": {"name": "Outdoor afternoon-work demo", "base": 0.56, "amplitude": 0.16, "exposure": 0.86, "start": 7, "end": 19},
    "Al Dhafra": {"name": "Remote rotating-shift demo", "base": 0.58, "amplitude": 0.10, "exposure": 0.80, "start": 5, "end": 20},
}

ABU_DHABI_AREAS = {
    "Abu Dhabi Island": (24.4539, 54.3773),
    "Khalifa City": (24.4259, 54.6050),
    "Mussafah": (24.3580, 54.4820),
    "Al Ain": (24.1917, 55.7606),
    "Al Dhafra": (23.6575, 53.7050),
}

def fetch_open_meteo(area: str, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo Archive API."""
    lat, lon = ABU_DHABI_AREAS[area]
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Dubai"
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly or not hourly.get("time"):
        raise ValueError("Open-Meteo returned no hourly observations.")
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly.get("temperature_2m"),
        "relative_humidity_pct": hourly.get("relative_humidity_2m"),
        "wind_speed_kmh": hourly.get("wind_speed_10m"),
    })
    df["area"] = area
    # Approximate wet-bulb temperature using Stull's formula; suitable for
    # screening/visualization, not a certified occupational WBGT measurement.
    t = df["temperature_c"].astype(float)
    rh = df["relative_humidity_pct"].astype(float).clip(1, 100)
    df["wet_bulb_approx_c"] = (
        t*np.arctan(0.151977*np.sqrt(rh+8.313659))
        + np.arctan(t+rh) - np.arctan(rh-1.676331)
        + 0.00391838*(rh**1.5)*np.arctan(0.023101*rh)
        - 4.686035
    )
    return df

def make_synthetic_data(days: int = 60, seed: int = 17, areas=None) -> pd.DataFrame:
    """Create clearly synthetic hourly operational signals for a reproducible demo."""
    rng = np.random.default_rng(seed)
    areas = areas or list(ABU_DHABI_AREAS.keys())
    end = pd.Timestamp.now(tz="Asia/Dubai").floor("h").tz_localize(None)
    times = pd.date_range(end=end, periods=days*24, freq="h")
    rows = []
    for ai, area in enumerate(areas):
        phase = ai * 0.35
        for i, ts in enumerate(times):
            hour = ts.hour
            day_idx = i / 24
            seasonal = 0.7*np.sin(2*np.pi*day_idx/30 + phase)
            diurnal = 5.2*np.sin(2*np.pi*(hour-8)/24)
            temp = 36.5 + seasonal + diurnal + rng.normal(0, 1.15)
            humidity = np.clip(48 - 0.55*(temp-30) + 9*np.sin(2*np.pi*(hour+2)/24) + rng.normal(0, 6), 12, 92)
            wind = np.clip(12 + rng.normal(0, 4), 0, 35)
            # Approximate wet-bulb temperature, used as a derived environmental feature.
            rh = np.clip(humidity, 1, 100)
            tw = (temp*np.arctan(0.151977*np.sqrt(rh+8.313659))
                  + np.arctan(temp+rh) - np.arctan(rh-1.676331)
                  + 0.00391838*(rh**1.5)*np.arctan(0.023101*rh) - 4.686035)
            profile = DEMO_OPERATIONAL_PROFILES.get(area, {
                "name": "Generic synthetic shift demo", "base": 0.55,
                "amplitude": 0.10, "exposure": 0.80, "start": 6, "end": 18
            })
            # Distinct scenario curves make geographic comparisons testable.
            # All operational values remain synthetic and visibly labelled.
            shift_phase = 2*np.pi*(hour - 8)/24
            workload = np.clip(
                profile["base"] + profile["amplitude"]*np.sin(shift_phase)
                + rng.normal(0, .055), .10, 1.0
            )
            in_shift = profile["start"] <= hour <= profile["end"]
            shift_exposure = np.clip(
                workload * (profile["exposure"] if in_shift else 0.22), 0, 1
            )
            # Baseline service indicators are synthetic counts per hour.
            first_aid = max(0, rng.poisson(0.28 + max(tw-25, 0)*0.025))
            clinic = max(0, rng.poisson(0.45 + max(tw-25, 0)*0.035))
            absence = max(0, rng.poisson(0.12 + max(tw-25, 0)*0.012))
            rows.append((ts, area, temp, humidity, wind, tw, workload, shift_exposure,
                         first_aid, clinic, absence, 0, profile["name"]))
    df = pd.DataFrame(rows, columns=[
        "timestamp","area","temperature_c","relative_humidity_pct","wind_speed_kmh",
        "wet_bulb_approx_c","workload_index","shift_exposure_index",
        "first_aid_reports","heat_related_clinic_visits","absence_reports","injected_event",
        "demo_operational_profile"
    ])
    # Inject two multi-signal episodes into a subset of locations, for demo/back-test only.
    for area, start_day, duration, intensity in [
        (areas[1] if len(areas)>1 else areas[0], days-17, 36, 1.0),
        (areas[3] if len(areas)>3 else areas[-1], days-7, 24, 0.8),
    ]:
        mask = (df["area"] == area) & (df["timestamp"] >= end - pd.Timedelta(days=days-start_day)) 
        mask &= (df["timestamp"] < end - pd.Timedelta(days=days-start_day) + pd.Timedelta(hours=duration))
        df.loc[mask, "temperature_c"] += 2.0*intensity
        df.loc[mask, "wet_bulb_approx_c"] += 1.5*intensity
        df.loc[mask, "shift_exposure_index"] = np.clip(df.loc[mask, "shift_exposure_index"] + .22*intensity, 0, 1)
        df.loc[mask, "first_aid_reports"] += rng.poisson(1.2*intensity, mask.sum())
        df.loc[mask, "heat_related_clinic_visits"] += rng.poisson(1.0*intensity, mask.sum())
        df.loc[mask, "absence_reports"] += rng.poisson(.6*intensity, mask.sum())
        df.loc[mask, "injected_event"] = 1
    return df

def get_demo_data(days: int = 60, use_live_weather: bool = False) -> tuple[pd.DataFrame, str]:
    """Return joined demo data and provenance label. Operational data remain synthetic."""
    areas = list(ABU_DHABI_AREAS.keys())
    synthetic = make_synthetic_data(days=days, areas=areas)
    if not use_live_weather:
        return synthetic, "Synthetic demonstration dataset (all signals simulated)"
    # Fetch public weather for each area; keep synthetic workplace/health signals separate.
    start = synthetic["timestamp"].min().date()
    end = synthetic["timestamp"].max().date()
    weather_parts = []
    errors = []
    for area in areas:
        try:
            weather_parts.append(fetch_open_meteo(area, start, end))
        except Exception as exc:
            errors.append(f"{area}: {exc}")
    if not weather_parts:
        return synthetic, "Live weather unavailable; using synthetic demonstration dataset"
    weather = pd.concat(weather_parts, ignore_index=True)
    # Hourly alignment: API observations replace synthetic environmental fields where available.
    merged = synthetic.drop(columns=["temperature_c","relative_humidity_pct","wind_speed_kmh","wet_bulb_approx_c"]).merge(
        weather, on=["timestamp","area"], how="left"
    )
    for col in ["temperature_c","relative_humidity_pct","wind_speed_kmh","wet_bulb_approx_c"]:
        merged[col] = merged[col].fillna(synthetic[col])
    provenance = "Public historical weather from Open-Meteo + synthetic workplace and health-service signals"
    if errors:
        provenance += " (some locations fell back to synthetic weather)"
    return merged, provenance


def fetch_open_meteo_forecast(area: str, forecast_hours: int = 48) -> pd.DataFrame:
    """Fetch public hourly forecast weather for one Abu Dhabi monitoring area."""
    if area not in ABU_DHABI_AREAS:
        raise KeyError(f"Unknown monitoring area: {area}")
    lat, lon = ABU_DHABI_AREAS[area]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,wind_speed_10m,"
            "wet_bulb_temperature_2m"
        ),
        "forecast_days": 2,
        "timezone": "Asia/Dubai",
        "wind_speed_unit": "kmh"
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly or not hourly.get("time"):
        raise ValueError(f"Open-Meteo returned no forecast for {area}.")
    result = pd.DataFrame({
        "forecast_timestamp": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly.get("temperature_2m"),
        "relative_humidity_pct": hourly.get("relative_humidity_2m"),
        "wind_speed_kmh": hourly.get("wind_speed_10m"),
        "wet_bulb_forecast_c": hourly.get("wet_bulb_temperature_2m")
    })
    result["area"] = area
    result["weather_source"] = "Open-Meteo forecast API"
    result["forecast_timezone"] = payload.get("timezone", "Asia/Dubai")
    return result

def get_live_weather_forecast(areas=None, forecast_hours: int = 24) -> tuple[pd.DataFrame, list[str]]:
    """Fetch forecast weather for all requested areas; report per-area failures."""
    areas = areas or list(ABU_DHABI_AREAS.keys())
    parts, errors = [], []
    for area in areas:
        try:
            part = fetch_open_meteo_forecast(area)
            parts.append(part)
        except Exception as exc:
            errors.append(f"{area}: {exc}")
    if not parts:
        return pd.DataFrame(), errors
    forecast = pd.concat(parts, ignore_index=True)
    # Keep the next requested forecast horizon from the current Dubai-local hour.
    now = pd.Timestamp.now(tz="Asia/Dubai").tz_localize(None).floor("h")
    forecast = forecast[
        (forecast["forecast_timestamp"] >= now)
        & (forecast["forecast_timestamp"] <= now + pd.Timedelta(hours=forecast_hours))
    ].copy()
    return forecast, errors
