import pandas as pd
from pathlib import *

folder_path = Path(__file__).parent.parent.absolute()
path = folder_path / "data" / "day_ahead_electricity_prices"
files = path.glob("*.csv")
day_ahead_prices = []

for file in files:
    day_ahead_price = pd.read_csv(file)
    for i in range(len(day_ahead_price)):
        index = str(day_ahead_price.iloc[i, 0]).find('"')
        day_ahead_prices.append(float(str(day_ahead_price.iloc[i, 0])[index + 1 : -1]))

day_ahead_prices = pd.DataFrame(day_ahead_prices, columns=["Day_ahead_price"])
day_ahead_prices.to_csv(
    folder_path / "data" / "day_ahead_electricity_prices.csv",
    index=False,
    header=True,
)
