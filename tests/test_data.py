import birds.train_model as train


def test_schema():
    assert train.load_data("./data") is not None


def test_missing_values():
    pass


def test_distribution():
    pass
