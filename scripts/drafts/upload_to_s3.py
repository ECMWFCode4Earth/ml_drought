import boto3
import os
from pathlib import Path

BUCKET = "mantlelabs-vci-forecast"
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')


def upload_file(file_name, bucket=BUCKET, object_name=None):
    """Upload a file to an S3 bucket
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


if __name__ == "__main__":
    s3 = boto3.resource("s3")
    all_buckets = [bucket.name for bucket in s3.buckets.all()]

    assert BUCKET in all_buckets, f"{BUCKET} not found in {all_buckets}"
