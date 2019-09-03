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

try:
    import xesmf
except ImportError:
    # pretty much all the preprocessors rely on
    # xesmf for regridding. They'll all fail without it
    # so this is an easier solutuon then adding an xfail
    # to all the modules
    collect_ignore.append('preprocess')
