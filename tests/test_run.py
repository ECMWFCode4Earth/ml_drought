import pytest

from src import DictWithDefaults


class TestDictWithDefaults:

    # This default dict needs to be updated as the
    # actual default dict in pipeline_config/default.json
    # is updated

    default_dict = {"data": "data", "export": {"era5": [{"variable": "precipitation"}]}}

    def test_missing_key(self):

        config = {}
        default_dict = self.default_dict.copy()
        default_dict.pop('data', None)

        with pytest.raises(Exception) as exception_info:
            DictWithDefaults(config, default_dict)
        print(default_dict)

        error_message = 'data is not defined ' \
                        'in the user config or the default config. Try using ' \
                        'the default config in pipeline_config/default.json'
        assert error_message in error_message, f'Got unexpected error message: {exception_info}'
