from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm_models import Result, ConfusionMatrix, ConfusionMatrixNumber, ConfusionMatrixLabel, SelectedFeature, Arg


def store_results(accuracy, f_measure, adj_rand_score, silhouette, initial_n_features, final_n_features, start_time,
                  end_time, confusion_matrix,
                  args, selected_columns, result_name):
    """

    :type args: argparse.Namespace
    :type final_n_features: int
    :type initial_n_features: int
    :type result_name: str
    :type selected_columns: list
    :type end_time: datetime.datetime
    :type start_time: datetime.datetime
    :type silhouette: float
    :type adj_rand_score: float
    :type f_measure: float
    :type accuracy: float
    :type confusion_matrix: pandas.DataFrame
    """

    engine = create_engine('sqlite:///local.db', echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    result_entry = Result(
        name=result_name,
        start_time=start_time,
        end_time=end_time,
        accuracy=accuracy,
        f_measure=f_measure,
        adjusted_rand_score=adj_rand_score,
        silhouette=silhouette,
        initial_n_features=initial_n_features,
        final_n_features=final_n_features
    )
    session.add(result_entry)
    session.flush()

    args_entries = []
    for arg_name, arg_val in vars(args).items():
        arg_entry = Arg(
            result_id=result_entry.id,
            name=arg_name,
            value=arg_val
        )
        args_entries.append(arg_entry)
    session.add_all(args_entries)

    cm_entry = ConfusionMatrix(
        result_id=result_entry.id
    )
    session.add(cm_entry)
    session.flush()

    cm_matrix = confusion_matrix.values

    labels = confusion_matrix.index.values
    cm_numbers_entries = []
    cm_labels_entries = []

    for i, row in enumerate(cm_matrix):
        cm_label_entry = ConfusionMatrixLabel(
            confusion_matrix_id=cm_entry.id,
            row_column=i,
            label=labels[i]
        )
        cm_labels_entries.append(cm_label_entry)
        for j, val in enumerate(row):
            cm_number_entry = ConfusionMatrixNumber(
                confusion_matrix_id=cm_entry.id,
                value=val,
                row=i,
                column=j
            )
            cm_numbers_entries.append(cm_number_entry)
    session.add_all(cm_numbers_entries)
    session.add_all(cm_labels_entries)

    selected_features_entries = []
    for column in selected_columns:
        selected_features_entry = SelectedFeature(
            result_id=result_entry.id,
            column=column
        )
        selected_features_entries.append(selected_features_entry)
    session.add_all(selected_features_entries)

    session.commit()
    session.close()