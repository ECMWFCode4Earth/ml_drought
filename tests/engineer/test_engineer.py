from src.engineer import Engineer


class TestEngineer:

    def test_get_preprocessed(self, tmp_path):

        # setup
        interim_folder = tmp_path / 'interim'
        interim_folder.mkdir()

        expected_output = []
        for foldername in ['a_preprocessed', 'b_preprocessed']:
            (interim_folder / foldername).mkdir()

            # this file should be captured
            filename = interim_folder / f'{foldername}/hello.nc'
            filename.touch()
            expected_output.append(filename)

            # this file should not
            (interim_folder / f'{foldername}/boo').touch()

        # none of this should be captured
        (interim_folder / 'woops').mkdir()
        (interim_folder / 'woops/hi.nc').touch()

        engineer = Engineer(tmp_path)
        files = engineer._get_preprocessed_files()

        assert set(expected_output) == set(files), f'Did not retrieve expected files!'
