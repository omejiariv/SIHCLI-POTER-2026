def test_import_project():
    import importlib

    importlib.import_module("app")
    assert True
