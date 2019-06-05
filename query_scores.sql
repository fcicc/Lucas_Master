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