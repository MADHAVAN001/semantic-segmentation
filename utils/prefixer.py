def fetch_prefix(run_type="train"):
    if run_type == "train":
        prefix = "training"
    elif run_type == "validation":
        prefix = "validation"
    else:
        prefix = "test"

    return prefix
