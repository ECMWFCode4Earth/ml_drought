# Working with S3

These are commands to help with using S3

ALWAYS:
- use `--dryrun` first to confirm that it's going to do what you expect

## Moving files to S3

```bash
aws s3 cp --recursive 2m_temperature s3://mantlelabs-vci-forecast/
```

NOTE: you want to use the whole path

## Removing files on S3

Delete two files:
- "worksheet.xlsx"
- "purple.gif"

NOTE: need the `--exclude "*"` first, this means that we DON'T delete everything
NOTE: use `--dryrun` first to check what the command will do

```bash
aws s3 rm s3://x.y.z/ --recursive --dryrun --exclude "*" --include "purple.gif" --include "worksheet.xlsx"
```

Links:cd vo
[Remove Multiple Files](https://stackoverflow.com/questions/41733318/how-to-delete-multiple-files-in-s3-bucket-with-aws-cli)