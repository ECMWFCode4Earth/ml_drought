from collections import namedtuple


Region = namedtuple('Region', ['name', 'lonmin', 'lonmax', 'latmin', 'latmax'])


def get_kenya() -> Region:
    """This pipeline is focused on drought prediction in Kenya.
    This function allows Kenya's bounding box to be easily accessed
    by all exporters.
    """
    return Region(name='kenya', lonmin=33.501, lonmax=42.283,
                  latmin=-5.202, latmax=6.002)
