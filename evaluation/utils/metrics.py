import pandas as pd


def get_int_predictions(df: pd.DataFrame, prediction_column: str = "prediction"):
    """Extracts the first character of the prediction column and converts it to an integer."""
    return pd.to_numeric(df[prediction_column].astype(str).str[0], errors="coerce").fillna(-1).astype(int)


def hard_accuracy(
    df: pd.DataFrame, prediction_column: str = "prediction", expected_output_column: str = "expected_output"
):
    """Calculates Hard Accuracy as the proportion of correct predictions."""
    df["parsed_prediction"] = get_int_predictions(df, prediction_column)
    correct = (df[expected_output_column] == df["parsed_prediction"]).sum()
    return correct / len(df)


def hard_accuracy_for_label(
    df: pd.DataFrame,
    label: int = 0,
    prediction_column: str = "prediction",
    expected_output_column: str = "expected_output",
):
    """Calculates Hard Accuracy for a specific label."""
    df["parsed_prediction"] = get_int_predictions(df, prediction_column)
    df_subset = df[df[expected_output_column] == label]
    correct = (df_subset["parsed_prediction"] == label).sum()
    return correct / len(df_subset)


def soft_accuracy(
    df: pd.DataFrame,
    prediction_column: str = "prediction",
    expected_output_column: str = "expected_output",
    label_to_ignore: int = 1,
):
    """Calculates Soft Accuracy, where incorrect 'it's okay' (1) predictions are also counted as correct."""
    df["parsed_prediction"] = get_int_predictions(df, prediction_column)

    correct = (df[expected_output_column] == df["parsed_prediction"]).sum()
    soft_correct = (
        (df[expected_output_column] == label_to_ignore) & (df[expected_output_column] != df["parsed_prediction"])
    ).sum()

    return (correct + soft_correct) / len(df)
