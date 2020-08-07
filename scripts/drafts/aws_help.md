# Working with S3

These are commands to help with using S3

ALWAYS:
- use `--dryrun` first to confirm that it's going to do what you expect

## Listing files in S3

```bash
aws s3 ls s3://mantlelabs-vci-forecast/data/raw/
```

## Moving files to S3

```bash
aws s3 cp --dryrun --recursive 2m_temperature s3://mantlelabs-vci-forecast/data/raw/reanalysis-era5-land/2m_temperature
```

```bash
aws s3 sync --dryrun data/raw s3://mantlelabs-vci-forecast/data/raw/ --exclude "*" --include "*reanalysis-era5-land/*"
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