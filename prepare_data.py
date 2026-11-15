"""Build data/turnarounds.csv.gz, the one table the notebook uses.

    python prepare_data.py

Steps:
  1. Download BTS On-Time Performance for every month of 2023, 2024 and 2025
     (one full year each for training, validation and testing).
  2. Link each arriving flight to the same tail number's next departure from
     that airport. One linked pair = one "turnaround".
  3. Attach the latest NOAA hourly weather observation BEFORE the arrival.
  4. Keep a random sample and save it.

Every feature saved here is known when the inbound flight reaches the gate.
The next flight's actual departure is used only to build the label y.
Downloads are cached in data/raw, so a second run is fast.
"""
import io
import time
import zipfile
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd
import requests

SEED = 5622
YEARS = (2023, 2024, 2025)     # train, validation, test: every season in each
SAMPLE_ROWS = 450_000
DELAY_MIN = 15                 # next flight "delayed" = leaves 15+ minutes late
MIN_TURN, MAX_TURN = 15, 240   # scheduled minutes on the ground
MAX_STATION_KM = 15            # weather station must be this close to the airport
MAX_WEATHER_AGE = "3h"         # older observations count as missing

RAW = Path("data/raw")
OUT = Path("data/turnarounds.csv.gz")
BTS_URL = ("https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_"
           "Performance_1987_present_{year}_{month}.zip")
NOAA = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly"
BTS_COLS = ["FlightDate", "Reporting_Airline", "Tail_Number", "Origin", "Dest",
            "CRSDepTime", "DepDelay", "ArrDelay", "CRSElapsedTime",
            "Cancelled", "Diverted", "Distance"]
WEATHER_COLS = ["wind_speed", "visibility", "precipitation"]


def download(url, tries=5):
    """Return the file's bytes, or None if it does not exist. Retries on network errors."""
    for attempt in range(tries):
        try:
            response = requests.get(url, timeout=300)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            print(f"retrying {url.split('/')[-1]} after: {error}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {url}")


# ---------------------------------------------------------------- 1. flights
def load_flights():
    months = []
    for year, month in [(y, m) for y in YEARS for m in range(1, 13)]:
        cache = RAW / f"bts_{year}_{month:02d}.parquet"
        if not cache.exists():
            print(f"downloading BTS {year}-{month:02d}")
            archive = zipfile.ZipFile(io.BytesIO(download(BTS_URL.format(year=year, month=month))))
            csv_name = [n for n in archive.namelist() if n.endswith(".csv")][0]
            cache.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(archive.open(csv_name), usecols=BTS_COLS, low_memory=False).to_parquet(cache)
        months.append(pd.read_parquet(cache, columns=BTS_COLS))
    return pd.concat(months, ignore_index=True)


def add_utc_times(flights, airports):
    """BTS clocks are local. Convert scheduled departure to UTC, then get the
    other times by adding the scheduled duration and the reported delays."""
    hhmm = flights["CRSDepTime"].astype(int)
    local = pd.to_datetime(flights["FlightDate"]) + pd.to_timedelta(hhmm // 100 * 60 + hhmm % 100, unit="m")
    zone = flights["Origin"].map({code: airports[code]["tz"] for code in airports})
    sched_dep = pd.Series(pd.NaT, index=flights.index)
    for tz, rows in zone.groupby(zone).groups.items():
        utc = local[rows].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        sched_dep[rows] = utc.dt.tz_convert("UTC").dt.tz_localize(None)
    sched_arr = sched_dep + pd.to_timedelta(flights["CRSElapsedTime"], unit="m")
    return flights.assign(
        sched_dep=sched_dep, sched_arr=sched_arr, dep_hour=hhmm // 100 % 24,
        actual_dep=sched_dep + pd.to_timedelta(flights["DepDelay"], unit="m"),
        actual_arr=sched_arr + pd.to_timedelta(flights["ArrDelay"], unit="m"))


# ------------------------------------------------------------ 2. turnarounds
def link_turnarounds(flights):
    """Sort each aircraft's flights in time; the next row is its next flight."""
    f = flights.dropna(subset=["Tail_Number", "sched_dep"])
    f = f.sort_values(["Tail_Number", "sched_dep"]).reset_index(drop=True)
    nxt = f.shift(-1)
    turn_min = (nxt["sched_dep"] - f["sched_arr"]).dt.total_seconds() / 60
    keep = ((f["Tail_Number"] == nxt["Tail_Number"])     # same aircraft
            & (f["Dest"] == nxt["Origin"])                # leaves from where it landed
            & (f["Cancelled"] == 0) & (f["Diverted"] == 0) & f["ArrDelay"].notna()
            & (nxt["Cancelled"] == 0) & nxt["DepDelay"].notna()
            & turn_min.between(MIN_TURN, MAX_TURN)
            & (nxt["actual_dep"] >= f["actual_arr"]))     # cannot leave before arriving
    print(f"{keep.sum():,} turnarounds from {len(flights):,} flights")
    leg_of_day = f.groupby(["Tail_Number", f["sched_dep"].dt.floor("D")]).cumcount() + 1
    turns = pd.DataFrame({
        "next_flight_date": nxt["FlightDate"],
        "carrier": f["Reporting_Airline"],
        "airport": f["Dest"],
        "arrival_time_utc": f["actual_arr"],              # the moment we predict
        "inbound_arr_delay": f["ArrDelay"],
        "sched_turn_min": turn_min,
        "slack_min": turn_min - f["ArrDelay"],            # ground time actually left
        "next_dep_hour": nxt["dep_hour"],
        "day_of_week": pd.to_datetime(nxt["FlightDate"]).dt.dayofweek,
        "leg_of_day": leg_of_day,
        "next_distance": nxt["Distance"],
        "y": (nxt["DepDelay"] >= DELAY_MIN).astype(int),
    })
    return turns[keep]


# ---------------------------------------------------------------- 3. weather
def nearest_stations(airports, codes):
    """For each airport, NOAA station ids to try: matching ICAO code first, then nearest."""
    cache = RAW / "ghcnh-station-list.csv"
    if not cache.exists():
        cache.write_bytes(download(f"{NOAA}/doc/ghcnh-station-list.csv"))
    stations = pd.read_csv(cache, usecols=["GHCN_ID", "LATITUDE", "LONGITUDE", "ICAO"]).dropna(
        subset=["LATITUDE", "LONGITUDE"])
    choices = {}
    for code in codes:
        lat, lon = airports[code]["lat"], airports[code]["lon"]
        km = 111 * np.hypot(stations["LATITUDE"] - lat,
                            (stations["LONGITUDE"] - lon) * np.cos(np.radians(lat)))
        near = stations.assign(km=km, match=stations["ICAO"] == airports[code]["icao"])
        near = near[near["km"] <= MAX_STATION_KM].sort_values(["match", "km"], ascending=[False, True])
        choices[code] = list(near["GHCN_ID"].head(3))
    return choices


def load_weather(airports, codes):
    frames = []
    stations = nearest_stations(airports, codes)
    for code, year in [(c, y) for c in codes for y in YEARS]:
        cache = RAW / f"weather_{code}_{year}.parquet"
        if not cache.exists():
            for station in stations[code]:
                raw = download(f"{NOAA}/access/by-year/{year}/parquet/GHCNh_{station}_{year}.parquet")
                if raw is None:
                    continue
                obs = pd.read_parquet(io.BytesIO(raw), columns=["DATE", *WEATHER_COLS])
                if len(obs) > 1000:                       # skip stations that barely report
                    obs.to_parquet(cache)
                    break
        if cache.exists():
            frames.append(pd.read_parquet(cache).assign(airport=code))
    weather = pd.concat(frames, ignore_index=True)
    for col in WEATHER_COLS:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather["precipitation"] = weather["precipitation"].fillna(0)   # blank = none reported
    weather["obs_time_utc"] = pd.to_datetime(weather["DATE"], errors="coerce").astype("datetime64[ns]")
    return weather.dropna(subset=["obs_time_utc"]).sort_values("obs_time_utc")


def add_weather(turns, weather):
    """Latest observation strictly before the arrival, at the turnaround airport."""
    turns = turns.assign(arrival_time_utc=turns["arrival_time_utc"].astype("datetime64[ns]"))
    merged = pd.merge_asof(
        turns.sort_values("arrival_time_utc"),
        weather[["airport", "obs_time_utc", *WEATHER_COLS]],
        left_on="arrival_time_utc", right_on="obs_time_utc", by="airport",
        direction="backward", allow_exact_matches=False, tolerance=pd.Timedelta(MAX_WEATHER_AGE))
    merged["weather_missing"] = merged["obs_time_utc"].isna().astype(int)
    for col in WEATHER_COLS:
        merged[col] = merged[col].fillna(merged[col].median())
    return merged.drop(columns="obs_time_utc")


# -------------------------------------------------------------------- 4. run
def main():
    airports = airportsdata.load("IATA")
    flights = load_flights()
    flights = flights[flights["Origin"].isin(airports) & flights["Dest"].isin(airports)]
    turns = link_turnarounds(add_utc_times(flights, airports))
    turns = turns.sample(n=SAMPLE_ROWS, random_state=SEED)
    turns = add_weather(turns, load_weather(airports, sorted(turns["airport"].unique())))
    turns = turns.sort_values(["next_flight_date", "arrival_time_utc"]).drop(columns="arrival_time_utc")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    turns.to_csv(OUT, index=False)
    print(f"saved {len(turns):,} rows to {OUT}; delay rate {turns['y'].mean():.1%}; "
          f"weather missing {turns['weather_missing'].mean():.2%}")


if __name__ == "__main__":
    main()
