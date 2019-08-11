from pathlib import Path
import sys
sys.path.append('..')

from src.analysis import RegionAnalysis


def run_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    analyzer = RegionAnalysis(data_dir)
    analyzer.analyze()


if __name__ == '__main__':
    run_region_analysis()
