from package.orm_models import Result, ConfusionMatrix, ConfusionMatrixNumber, ConfusionMatrixLabel, SelectedFeature, \
    Arg, \
    local_create_session, ClusterLabel, Score


def store_results(scores, initial_n_features, final_n_features, start_time,
                  end_time, execution_interval, confusion_matrix, args, selected_columns, ga_metrics, cluster_labels):
    """

    :type scores: dict
    :type ga_metrics: pandas.DataFrame
    :type args: argparse.Namespace
    :type final_n_features: int
    :type initial_n_features: int
    :type selected_columns: list
    :type end_time: datetime.datetime
    :type start_time: datetime.datetime
    :type execution_interval: datetime.timedelta
    :type confusion_matrix: pandas.DataFrame
    """

    result_name = args.experiment_name
    db_file = args.db_file

    session = local_create_session(db_file)

    result_entry = Result(
        name=result_name,
        start_time=start_time,
        end_time=end_time,
        execution_interval=execution_interval,
        initial_n_features=int(initial_n_features),
        final_n_features=int(final_n_features),
        individual_evaluations=ga_metrics
    )
    session.add(result_entry)
    session.flush()

    args_entries = []
    for arg_name, arg_val in vars(args).items():
        arg_entry = Arg(
            result_id=result_entry.id,
            name=arg_name,
            value=str(arg_val)
        )
        args_entries.append(arg_entry)
    session.bulk_save_objects(args_entries)

    scores_entries = []
    for score_name, score_val in scores.items():
        score_entry = Score(
            result_id=result_entry.id,
            name=score_name,
            value=str(score_val[0])
        )
        scores_entries.append(score_entry)
    session.bulk_save_objects(scores_entries)

    cm_entry = ConfusionMatrix(
        result_id=result_entry.id
    )
    session.add(cm_entry)
    session.flush()

    # cm_matrix = confusion_matrix.values

    # labels = confusion_matrix.index.values
    # cm_numbers_entries = []
    # cm_labels_entries = []

    # for i, row in enumerate(cm_matrix):
    #     cm_label_entry = ConfusionMatrixLabel(
    #         confusion_matrix_id=cm_entry.id,
    #         row_column=int(i),
    #         label=labels[i]
    #     )
    #     cm_labels_entries.append(cm_label_entry)
    #     for j, val in enumerate(row):
    #         cm_number_entry = ConfusionMatrixNumber(
    #             confusion_matrix_id=cm_entry.id,
    #             value=float(val),
    #             row=int(i),
    #             column=int(j)
    #         )
    #         cm_numbers_entries.append(cm_number_entry)
    # session.bulk_save_objects(cm_numbers_entries)
    # session.bulk_save_objects(cm_labels_entries)

    selected_features_entries = []
    for column in selected_columns:
        selected_features_entry = SelectedFeature(
            result_id=result_entry.id,
            column=column
        )
        selected_features_entries.append(selected_features_entry)
    session.bulk_save_objects(selected_features_entries)

    cluster_labels_entries = []
    for label in cluster_labels:
        cluster_label_entry = ClusterLabel(
            result_id=result_entry.id,
            label=str(label)
        )
        cluster_labels_entries.append(cluster_label_entry)
    session.bulk_save_objects(cluster_labels_entries)

    session.commit()
    result_entry_id = result_entry.id
    session.close()

    return result_entry_id
