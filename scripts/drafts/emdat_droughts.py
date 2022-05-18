import pandas as pd 
from pathlib import Path


if __name__ == "__main__":
    # download data for climatological / drought from EMDAT
    # https://public.emdat.be/data (requires registration)

    data_dir = Path(".").home() / "Downloads"

    # read data
    df = pd.read_excel(list(data_dir.glob("emdat*.xl*"))[0], skiprows=6)

    # 2003 EU, 2003 EU, 2005 Amazon, 2010 Russian, 2010 EA, 2010 EA, 2011 Ethiopia, 2012 California, 2015 SA
    DISASTER_CODES = ["2003-9784-BIH", "2003-9784-HUN", "2005-9569-BRA", "2010-9318-RUS", "2010-9082-KEN", "2010-9082-SOM", "2011-9663-ETH", "2012-9489-USA", "2015-9500-ZAF"]

    df = df.loc[df["Dis No"].isin(DISASTER_CODES)]