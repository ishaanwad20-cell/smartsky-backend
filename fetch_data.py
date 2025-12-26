import requests
import csv
from datetime import datetime
import sys

API_BASE = "http://127.0.0.1:8000/api/planet"

def fetch_and_save(planet="mars", days=30, lat=None, lon=None, outfn="mars_ephemeris.csv"):
    params = {"days": days}
    if lat is not None and lon is not None:
        params["lat"] = lat
        params["lon"] = lon

    url = f"{API_BASE}/{planet}"
    print(f"Requesting: {url}  params={params}")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # normalize keys into CSV columns
    headers = ["date","mode","ra_hours","dec_degrees","distance_au","altitude_deg","azimuth_deg","distance_km"]
    with open(outfn, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for item in data:
            row = {h: None for h in headers}
            row["date"] = item.get("date")
            row["mode"] = item.get("mode")
            if item.get("mode") == "radec":
                row["ra_hours"] = item.get("ra_hours")
                row["dec_degrees"] = item.get("dec_degrees")
                row["distance_au"] = item.get("distance_au")
            else:
                row["altitude_deg"] = item.get("altitude_deg")
                row["azimuth_deg"] = item.get("azimuth_deg")
                row["distance_km"] = item.get("distance_km")
            writer.writerow(row)
    print(f"Saved {outfn} ({len(data)} rows)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet", default="mars")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    outfn = args.out or f"{args.planet}_ephemeris_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    fetch_and_save(planet=args.planet, days=args.days, lat=args.lat, lon=args.lon, outfn=outfn)
