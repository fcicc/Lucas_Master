-- QUERY SCORE UTILS --

-- Query Average Scores ------------------------------------------------------------------------------------------------
SELECT AVG(score_accuracy.value), AVG(score_siulhouette.value), arg_experiment_name.value, AVG(strftime('%s',result.end_time) - strftime('%s',result.start_time)), arg_scenario.value
FROM result
	INNER JOIN score AS score_accuracy ON score_accuracy.result_id=result.id AND score_accuracy.name='accuracy'
	INNER JOIN score AS score_siulhouette ON score_siulhouette.result_id=result.id AND score_siulhouette.name='silhouette_sklearn'
	INNER JOIN arg AS arg_experiment_name ON arg_experiment_name.result_id=result.id AND arg_experiment_name.name='experiment_name'
	INNER JOIN arg AS arg_scenario ON arg_scenario.result_id=result.id AND arg_scenario.name='scenario'
GROUP BY result.name;


-- Query Update Scores -------------------------------------------------------------------------------------------------
SELECT AVG(score_accuracy.value), AVG(score_siulhouette.value), arg_experiment_name.value, AVG(strftime('%s',result.end_time) - strftime('%s',result.start_time)), arg_scenario.value
FROM result
	INNER JOIN score AS score_accuracy ON score_accuracy.result_id=result.id AND score_accuracy.name='accuracy'
	INNER JOIN score AS score_siulhouette ON score_siulhouette.result_id=result.id AND score_siulhouette.name='silhouette_sklearn'
	INNER JOIN arg AS arg_experiment_name ON arg_experiment_name.result_id=result.id AND arg_experiment_name.name='experiment_name'
	INNER JOIN arg AS arg_scenario ON arg_scenario.result_id=result.id AND arg_scenario.name='scenario'
GROUP BY result.name;

-- Query scores by dataset and scenario -------------------------------------------------------------------------------------------------
SELECT score_accuracy.value AS "ACCURACY SCORE",
       arg_cluster.value    AS "CLUSTER METHOD",
       arg_dataset.value    AS "DATASET",
       arg_scenario.value   AS "SCENARIO"
FROM result
         INNER JOIN arg AS arg_cluster ON arg_cluster.result_id = result.id AND arg_cluster.name = 'cluster_algorithm'
         INNER JOIN score AS score_accuracy ON score_accuracy.result_id = result.id AND score_accuracy.name = 'accuracy'
         INNER JOIN arg AS arg_dataset ON arg_dataset.result_id = result.id AND arg_dataset.name = 'input_file'
         INNER JOIN arg AS arg_scenario ON arg_scenario.result_id = result.id AND arg_scenario.name = 'scenario';

-- Query scores for results_1 -------------------------------------------------------------------------------------------------
SELECT score_accuracy.value AS "ACCURACY SCORE",
       arg_dataset.value    AS "DATASET",
       arg_scenario.value   AS "SCENARIO"
FROM result
         INNER JOIN score AS score_accuracy ON score_accuracy.result_id = result.id AND score_accuracy.name = 'accuracy'
         INNER JOIN arg AS arg_dataset ON arg_dataset.result_id = result.id AND arg_dataset.name = 'input_file'
         INNER JOIN arg AS arg_scenario ON arg_scenario.result_id = result.id AND arg_scenario.name = 'scenario';

-- Query scores for results_2 -------------------------------------------------------------------------------------------------
SELECT score_accuracy.value     AS "ACCURACY SCORE",
       arg_cluster.value        AS "CLUSTER METHOD",
       arg_dataset.value        AS "DATASET",
       arg_scenario.value       AS "SCENARIO",
       arg_fitness_metric.value AS "FITNESS METRIC"
FROM result
         INNER JOIN arg AS arg_cluster ON arg_cluster.result_id = result.id AND arg_cluster.name = 'cluster_algorithm'
         INNER JOIN score AS score_accuracy ON score_accuracy.result_id = result.id AND score_accuracy.name = 'accuracy'
         INNER JOIN arg AS arg_dataset ON arg_dataset.result_id = result.id AND arg_dataset.name = 'input_file'
         INNER JOIN arg AS arg_scenario ON arg_scenario.result_id = result.id AND arg_scenario.name = 'scenario'
         INNER JOIN arg AS arg_fitness_metric
                    ON arg_fitness_metric.result_id = result.id AND arg_fitness_metric.name = 'fitness_metric';

-- Query scores for results_4_1 -------------------------------------------------------------------------------------------------
SELECT score_accuracy.value        AS "ACCURACY SCORE",
       arg_cluster.value           AS "CLUSTER METHOD",
       arg_dataset.value           AS "DATASET",
       arg_scenario.value          AS "SCENARIO",
       COUNT(selected_features.id) AS "NUMBER OF SELECTED FEATURES"
FROM result
         INNER JOIN arg AS arg_cluster ON arg_cluster.result_id = result.id AND arg_cluster.name = 'cluster_algorithm'
         INNER JOIN score AS score_accuracy ON score_accuracy.result_id = result.id AND score_accuracy.name = 'accuracy'
         INNER JOIN arg AS arg_dataset ON arg_dataset.result_id = result.id AND arg_dataset.name = 'input_file'
         INNER JOIN arg AS arg_scenario ON arg_scenario.result_id = result.id AND arg_scenario.name = 'scenario'
         INNER JOIN selected_feature AS selected_features ON selected_features.result_id = result.id
GROUP BY result.id