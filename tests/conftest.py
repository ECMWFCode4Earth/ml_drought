collect_ignore = []
try:
    import boto3
    from moto import mock_s3
except ImportError:
    collect_ignore.append('exporters/test_planetOS.py')

try:
    import paramiko
except ImportError:
    collect_ignore.append('exporters/test_gleam.py')
