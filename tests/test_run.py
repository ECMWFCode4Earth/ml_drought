import pytest

from src import Run, DictWithDefaults


class TestDictWithDefaults:

    # This default dict needs to be updated as the
    # actual default dict in pipeline_config/default.json
    # is updated

    default_dict = {
        "data": "data",
        "export": {"era5": [{"variable": "precipitation"}]},
        "preprocess": ["vhi"],
        "engineer": {"init_args": {"process_static": True}},
    }

    def test_missing_key(self):

        config = {}
        default_dict = self.default_dict.copy()
        default_dict.pop("data", None)

        with pytest.raises(Exception) as exception_info:
            DictWithDefaults(config, default_dict)

        error_message = (
            "data is not defined "
            "in the user config or the default config. Try using "
            "the default config in pipeline_config/(minimal, full).json"
        )
        assert error_message in str(
            exception_info
        ), f"Got unexpected error message: {exception_info}"

    def test_user_defined_priority(self):

        user_config = {"data": "user_data"}

        dict_with_defaults = DictWithDefaults(user_config, self.default_dict)

        assert (
            dict_with_defaults["data"] == user_config["data"]
        ), f'Expected data to be {user_config["data"]}, got {dict_with_defaults["data"]}'


class TestRun:
    def test_dataset_assertion(self, tmp_path):

        runtask = Run(tmp_path)

        export_dict = {"era42": ["towels"]}
        with pytest.raises(Exception) as exception_info:
            runtask.export(export_dict)
        error_message_contains = "is not supported! Supported datasets are"
        assert error_message_contains in str(
            exception_info
        ), f"Got unexpected error message: {exception_info}"

    def test_dataset_values_assertion(self, tmp_path):
        runtask = Run(tmp_path)

        export_dict = {"era5": "not a list"}
        with pytest.raises(Exception) as exception_info:
            runtask.export(export_dict)
        error_message_contains = "values to be a list. Got"
        assert error_message_contains in str(
            exception_info
        ), f"Got unexpected error message: {exception_info}"
