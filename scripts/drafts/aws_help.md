# Working with S3

These are commands to help with using S3

ALWAYS:
- use `--dryrun` first to confirm that it's going to do what you expect

## Listing files in S3

```bash
aws s3 ls s3://mantlelabs-vci-forecast/data/raw/

aws s3 ls s3://mantlelabs-vci-forhtop
ecast/data/raw/ --region eu-central-1
```

## Moving files to S3

```bash
aws s3 cp --dryrun --recursive --region eu-central-1

aws s3 cp --dryrun --recursive 2m_temperature s3://mantlelabs-vci-forecast/data/raw/reanalysis-era5-land/2m_temperature

aws s3 cp --dryrun --recursive seasonal-monthly-single-levels/ s3://mantlelabs-vci-forecast/data/raw/seasonal-monthly-single-levels/ --region eu-central-1
```

```bash
aws s3 sync --dryrun data/raw s3://mantlelabs-vci-forecast/data/raw/ --exclude "*" --include "*reanalysis-era5-land/*"
aws s3 sync --dryrun data/raw s3://mantlelabs-vci-forecast/data/raw/ --exclude "*" --include "*reanalysis-era5-land-monthly*"
```

NOTE: you want to use the whole path (will iteratively create the dirs required)

## Removing files on S3

Delete two files:
- "worksheet.xlsx"
- "purple.gif"

NOTE: need the `--exclude "*"` first, this means that we DON'T delete everything
NOTE: use `--dryrun` first to check what the command will do

```bash
aws s3 rm s3://x.y.z/ --recursive --dryrun --exclude "*" --include "purple.gif" --include "worksheet.xlsx"

aws s3 rm --region eu-central-1 --dryrun s3://mantlelabs-vci-forecast/data/raw/esa_cci_landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992_2015-v2.0.7b.nc.zip --dryrun

--exclude "*" --include "purple.gif" --include "worksheet.xlsx"

```

Links:
[Remove Multiple Files](https://stackoverflow.com/questions/41733318/how-to-delete-multiple-files-in-s3-bucket-with-aws-cli)

## Reading data from S3

```bash
```

# Working with Mantle-Utils

```bash
git clone https://github.com/mantlelabs/mantle-utils.git

cd mantle-utils/mantle_utils
conda activate ml
python setup.py install
```


```python
import mantle_utils
```

# Getting Conda up and Runing
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# EXPORT from elsewhere
# conda env export | grep -v "^prefix: " > environment.yml

# CREATE new env ('ml')
conda env create -f environment.yml
```