def test_pkg_installation():
    error_msg = "HousePricePrediction Package is not installed correctly"
    try:
        import HousePricePrediction
    except Exception as e:
        assert False, f"Error: {e}, " + error_msg
