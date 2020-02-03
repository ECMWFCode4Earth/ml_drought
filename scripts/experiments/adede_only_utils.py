import sys

sys.path.append("../..")

from scripts.utils import _rename_directory, get_data_path


def rename_dirs():
    data_path = get_data_path()

    # INTERIM
    if (data_path / "interim_adede_only").exists() and (data_path / "interim").exists():
        # move interim -> interim_
        # move interim_adede -> interim
        print("Moving data/interim -> data/interim_")
        _rename_directory(
            from_path=data_path / "interim",
            to_path=data_path / "interim_",
            with_datetime=False,
        )

        print("Moving data/interim_adede_only -> data/interim")
        _rename_directory(
            from_path=data_path / "interim_adede_only",
            to_path=data_path / "interim",
            with_datetime=False,
        )
    elif (
        not (data_path / "interim_adede_only").exists()
        and (data_path / "interim_").exists()
    ):
        # move interim_adede -> interim
        print("Moving data/interim_adede_only -> data/interim")
        _rename_directory(
            from_path=data_path / "interim_adede_only",
            to_path=data_path / "interim",
            with_datetime=False,
        )

    # check that correct dirs created
    assert not (data_path / "interim_adede_only").exists()
    assert (data_path / "interim").exists()
    assert (data_path / "interim_").exists()

    # FEATURES
    if (data_path / "features" / "one_month_forecast").exists():
        print(
            "Moving data/features/one_month_forecast -> data/features/one_month_forecast_"
        )
        _rename_directory(
            from_path=data_path / "features/one_month_forecast",
            to_path=data_path / "features/one_month_forecast_",
            with_datetime=False,
        )

    assert not (data_path / "features" / "one_month_forecast").exists()


def revert_interim_dirs():
    data_path = get_data_path()
    # INTERIM
    print("Moving data/interim -> data/interim_adede")
    _rename_directory(
        from_path=data_path / "interim",
        to_path=data_path / f"interim_adede_only",
        with_datetime=False,
    )
    print("Moving data/interim_ -> data/interim")
    _rename_directory(
        from_path=data_path / "interim_",
        to_path=data_path / "interim",
        with_datetime=False,
    )


def revert_interim_dirs():
    data_path = get_data_path()
    # INTERIM
    print("Moving data/interim -> data/interim_adede")
    _rename_directory(
        from_path=data_path / "interim",
        to_path=data_path / f"interim_adede_only",
        with_datetime=False,
    )
    print("Moving data/interim_ -> data/interim")
    _rename_directory(
        from_path=data_path / "interim_",
        to_path=data_path / "interim",
        with_datetime=False,
    )
