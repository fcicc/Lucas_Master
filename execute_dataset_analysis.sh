#!/usr/bin/env bash
NGEN=100
POPSIZE=128
EVALLPROP=1
MINFEATURES=5
MAXFEATURES=10000

source activate mestrado

 for i in `seq 1 10`;
 do
 	echo "$i"
 	python ./main.py ../datasets/CampusBasin/results_ga_hc/dataset.csv "experiment_1" -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $MAXFEATURES #$(($j*5))
 done

 for i in `seq 1 10`;
 do
 	echo "$i"
 	python ./main.py ../datasets/MargemEquatorial/results_ga_hc/dataset.csv "experiment_2" -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $MAXFEATURES #$(($j*5))
 done

# for i in `seq 1 50`;
# do
# 	echo "$i"
# 	name=${i}
# 	python ./ga_kmeans.py ../datasets/CampusBasin/results_ga_kmeans/dataset.csv $name -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $MAXFEATURES #$(($j*5))
# done
#
# for i in `seq 1 50`;
# do
# 	echo "$i"
# 	name=${i}
# 	python ./ga_kmeans.py ../datasets/MargemEquatorial/results_ga_kmeans/dataset.csv $name -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $MAXFEATURES #$(($j*5))
# done
#
#for i in `seq 1 100`;
#do
#	echo "$i"
#	name=${i}
#	python ./ga_kmeans.py ../datasets/CampusBasin/results_kmeans/dataset.csv $name -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $MAXFEATURES #$(($j*5))
#done
#
#for i in `seq 1 100`;
#do
#	echo "$i"
#	name=${i}
#	python ./ga_kmeans.py ../datasets/MargemEquatorial/results_kmeans/dataset.csv $name -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $MAXFEATURES #$(($j*5))
#done
#
# for j in `seq 1 195`;
# do
# 	for i in `seq 1 10`;
# 	do
# 		name="$j_$i"
# 		echo $name
# 		python ./ga_hc.py ../datasets/MargemEquatorial/fixed_results_kmeans_ga/dataset.csv $name -e $EVALLPROP --num-gen $NGEN --pop-size $POPSIZE --min-features $MINFEATURES --max-features $(($j*5))
# 	done
# done
