import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def get_accuracy_precision_recall(cluster_file):
    """
    Meant to be use per day
    """

    clusters_df = pd.read_csv(cluster_file)

    # If there is no cluster at all for the day, the file will only contain header.
    # If this is the case, then we return everything as report as empty and averages as 0.

    # We notice that the header for csv file can either be day or timepoint.
    # Basically the name of first column is not always 'timepoint'.
    # So we try and make it flexible by first getting the header name of the first column
    header_first_col = list(clusters_df)[0]
    if len(clusters_df[header_first_col]) == 0:
        return {}, (0.00, 0.00, 0.00, 0), 0.00

    # This is needed for classification_report, because the label it produces is based on the order in true labels.
    clusters_df.sort_values("TrueLabel", inplace=True)
    true_labels = clusters_df['TrueLabel'].values
    predicted_labels = clusters_df['PredictedLabel'].values

    report_dict = classification_report(true_labels, predicted_labels, output_dict=True)

    weighted_average = report_dict['weighted avg']
    averages = (round(weighted_average['precision'], 2),
                round(weighted_average['recall'], 2),
                round(weighted_average['f1-score'], 2),
                round(weighted_average['support'], 2))
    accuracy = accuracy_score(true_labels, predicted_labels)

    return averages, accuracy

