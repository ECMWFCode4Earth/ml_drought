# Pipeline configurations

The pipeline is configured using `json` objects. 
The pipeline consists of the following steps (the default configuration is used as an example):

### 1. Export

The export step consists of downloading data from remote datasets to a local folder.

The format for exporter configurations is `{dataset: [list of variable configurations]}`.

The following exporters have been implemented:

#### 1.a. ERA5

Download ERA5 reanalysis data from the [climate data store](https://cds.climate.copernicus.eu/#!/home).
For more complete documentation on the inputs, see the [`ERA5 exporter docstring`](../src/exporters/cds.py)

```
"era5": [
         {
            "variable":"precipitation",
            "selection_request": {
                "year": List of ints between 1979 and 2019, default: all,
                "month": List of ints between 1 and 12, default: all,
                "day": List of ints between 1 and 31, default: all,
                "time": List of strings between "00:00" and "23:00" incrementing one hour, default: all
                                  }
            "granularity": string, {"hourly", "monthly"}, granularity of the data, default: "hourly"
            "show_api_request": boolean, whether to print the request made to the cdsapi, default: true
            "break_up": boolean, break up request into monthly chunks, default: true
            
         }]
```

#### 1.b VHI

Download vegetation health index data from the NOAA 
[Center for Satellite Applications and Research](https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/vh_browse.php).

For more complete documentation on the inputs, see the [`VHI exporter docstring`](../src/exporters/vhi.py)

```
"vhi": [
        {"years": List of ints between 1981 and 2019, default: all}
    ]
```
