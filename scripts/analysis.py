from pathlib import Path
import sys
sys.path.append('..')

from src.analysis import AdministrativeRegionAnalysis
from src.analysis import LandcoverRegionAnalysis


def run_administrative_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    assert [f for f in (data_path / 'features').glob('*/test/*/*.nc')] != [], \
        'There is no true data (has the pipeline been run?)'
    assert [f for f in (data_path / 'models').glob('*/*/*.nc')] != [], \
        'There is no model data (has the pipeline been run?)'
    assert [f for f in (data_path / 'analysis').glob('*/*.nc')] != [], \
        'There are no processed regions. '\
        'Has the pipeline been run?'

    analyzer = AdministrativeRegionAnalysis(data_dir)
    analyzer.analyze()
    print(analyzer.regional_mean_metrics)


def run_landcover_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    assert [f for f in (data_path / 'features').glob('*/test/*/*.nc')] != [], \
        'There is no true data (has the pipeline been run?)'
    assert [f for f in (data_path / 'models').glob('*/*/*.nc')] != [], \
        'There is no model data (has the pipeline been run?)'
    assert [
        f.name for f in (
            data_path / 'interim'/ 'static' /
            'esa_cci_landcover_preprocessed'
        ).glob('*')
    ] != [], \
        'There is no landcover data. '\
        'Has the pipeline been run?'

    analyzer = LandcoverRegionAnalysis(data_dir)
    analyzer.analyze()
    print(analyzer.regional_mean_metrics)


if __name__ == '__main__':
    run_administrative_region_analysis()
    run_landcover_region_analysis()
