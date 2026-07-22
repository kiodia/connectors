import connectors


def test_version_is_exposed():
    assert isinstance(connectors.__version__, str)
    assert connectors.__version__
