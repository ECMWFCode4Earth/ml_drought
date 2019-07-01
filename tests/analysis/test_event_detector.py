from src.analysis import EventDetector

from ..utils import _make_dataset


class TestEventDetector:
    @staticmethod
    def create_dummy_data(tmp_path):
        data_dir = tmp_path / 'data' / 'interim'
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        precip, _, _ = _make_dataset((30,30), variable_name='precip')
        precip.to_netcdf(data_dir / 'chirps_preprocessed' / 'chirps_kenya.nc')

    @staticmethod
    def create_test_consec_data(tmp_path):
        """
        create an array of 20 timesteps with 2 events below 1STD of the series
        event from indices 3:7 & 10:13
        """
        a = np.ones((20))*10
        a[3:7] = 0.2
        a[10:13] = 0.2
        p = np.repeat(a, 25).reshape(20, 5, 5)

        lat = np.arange(0,5)
        lon = np.arange(0,5)
        time = pd.date_range('2000-01-01', freq='M', periods=20)

        d = xr.Dataset(
            {'precip': (['time','lat','lon'], p)},
            coords={
                'lon': lon,
                'lat': lat,
                'time': time
            }
        )

        data_dir = tmp_path / 'data' / 'interim'
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        out_path = data_dir / 'chirps_preprocessed' / 'chirps_kenya.nc'
        d.to_netcdf(out_path)
        return out_path

    def test_initialise_event_detector(self, tmp_path):
        create_dummy_data(tmp_path)

        data_dir = tmp_path / 'data'
        precip_dir = data_dir / 'chirps_preprocessed' / 'chirps_kenya.nc'

        e = EventDetector(precip_dir)

        assert 'precip' in [v for v in e.ds.variables], f"Expected precip to be a variable.\
        Got: {[v for v in e.ds.variables]}"

        assert e.ds.precip.shape == (36, 30, 30), f"Expected shape of EventDetector.ds to be\ {(36, 30, 30)}. Got: {e.ds.precip.shape}"

    def test_number_of_consecutive_events(self, tmp_path):
        in_path = create_test_consec_data(tmp_path)

        e = EventDetector(in_path)
        e.detect(
            variable='precip', time_period='dayofyear', hilo='low', method='std'
        )

        pass
