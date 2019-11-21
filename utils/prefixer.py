def fetch_prefix(run_type="train"):
    if run_type == "train":
        prefix = "training"
    elif run_type == "validate":
        prefix = "validate"
    else:
        prefix = "test"

    return prefix
