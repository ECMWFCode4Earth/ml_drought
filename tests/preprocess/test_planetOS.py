from src.preprocess import PlanetOSPreprocessor


class TestPlanetOSPreprocessor:

    def test_init(self, tmp_path):

        PlanetOSPreprocessor(tmp_path)

        assert (tmp_path / 'interim/era5POS_preprocessed').exists()
        assert (tmp_path / 'interim/era5POS_interim').exists()
