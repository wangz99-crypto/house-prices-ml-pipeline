def metric_label_map():
    """
    把工程术语换成对外可读术语（但不撒谎）。
    """
    return {
        "cv_rmse": "Model quality score (lower is better)",
        "fold_rmse": "Quality checks (per split)",
        "n_splits": "Number of quality checks",
        "seed": "Reproducibility seed",
        "head200_sha256": "Data signature (preview hash)",
        "rows": "Rows",
        "cols": "Columns",
    }

def explain_quality_score():
    return (
        "This score summarizes how well the model performed during repeated validation checks. "
        "Lower is better. It is designed to be comparable across model versions."
    )

def explain_data_signature():
    return (
        "A lightweight signature of the dataset schema and a small preview sample. "
        "Used to ensure the model is evaluated and served with a compatible data format."
    )

def app_tagline():
    return (
        "A production-style machine learning system demo: live prediction, versioned models, "
        "reproducible runs, and transparent analysis."
    )
