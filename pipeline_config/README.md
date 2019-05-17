# Pipeline configurations

The pipeline is configured using `json` objects. 
The pipeline consists of the following steps (the default configuration is used as an example):

### 1. Export

The export step consists of downloading data from remote datasets to a local folder.

The format for exporter configurations is `{dataset: [list of variable configurations]}`.

```json
"export":{
  "era5":[
     {
        "variable":"precipitation"
     }
  ]
}
```
