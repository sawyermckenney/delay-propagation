# Predicting airline delay propagation

CSCI 5622 project. When an aircraft reaches the gate, should the airline's operations team flag it as at risk of making its **next** departure 15+ minutes late?

## Reproduce the headline number

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace delay_propagation.ipynb
```

Or open `delay_propagation.ipynb` in Colab or Jupyter and choose *Run all*. The seed is fixed (`SEED = 5622`). The headline number is the forest's test cost per 1,000 turnarounds, printed in section 6 beside the baseline.

## Files

| File | What it is |
|---|---|
| `delay_propagation.ipynb` | All modeling and evaluation |
| `data/turnarounds.csv.gz` | The data the notebook reads (450,000 turnarounds, included) |
| `prepare_data.py` | Rebuilds that file from the raw sources: `python prepare_data.py` (about 10 minutes of downloads the first time) |

## Data

- **Flights:** U.S. DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance, 2023 through 2025.
- **Weather:** NOAA Global Historical Climatology Network hourly (GHCNh), one station per airport within 15 km.

A turnaround is an arriving flight plus the same tail number's next departure from that airport, scheduled 15 to 240 minutes later. Every feature is known when the inbound reaches the gate.

## Design

| Item | Choice |
|---|---|
| Target | Next flight departs 15+ minutes late |
| Split | Chronological: train 2023, validation 2024, test 2025 (whole years, so every season is in each part) |
| Metric | Cost per 1,000 turnarounds (missed delay = 5, false alarm = 1) |
| Baseline | Slack rule: flag when scheduled turn time minus inbound arrival delay is at or below a cut-off |
| Models | Depth-3 decision tree (readable) and random forest (flexible) |
