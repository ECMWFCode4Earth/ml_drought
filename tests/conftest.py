collect_ignore = []
try:
    import boto3
    from moto import mock_s3
except ImportError:
    collect_ignore.append("exporters/test_planetOS.py")

try:
    import paramiko
except ImportError:
    collect_ignore.append("exporters/test_gleam.py")

try:
    import xesmf
except ImportError:
    # pretty much all the preprocessors rely on
    # xesmf for regridding. They'll all fail without it
    # so this is an easier solutuon then adding an xfail
    # to all the modules
    collect_ignore.append("preprocess")
    # this also screws up the run.py tests, since they
    # test for certain stdouts and get confused by
    # the exceptions
    collect_ignore.append("test_run.py")

try:
    import BeautifulSoup
except ImportError:
    collect_ignore.append("exporters/test_chirps.py")

try:
    import xclim
except ImportError:
    collect_ignore.append("analysis/test_event_detector.py")

try:
    import bottleneck
except ImportError:
    collect_ignore.append("analysis/indices")

try:
    import climate_indices
except ImportError:
    collect_ignore.append("analysis/indices/test_spi.py")

try:
    import cfgrib
except ImportError:
    collect_ignore.append("preprocess/test_s5.py")
