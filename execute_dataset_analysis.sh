NGEN=100
POPSIZE=600

python ./ga_hc.py ../datasets/polvo/full_scenario 10 -c --num-gen $NGEN --pop-size $POPSIZE
python ./ga_hc.py ../datasets/MargemEquatorial/full_scenario/ 20 -c --num-gen $NGEN --pop-size $POPSIZE

python ./ga_hc.py ../datasets/polvo/full_scenario 10 -c --perfect --num-gen $NGEN --pop-size $POPSIZE
python ./ga_hc.py ../datasets/MargemEquatorial/full_scenario/ 20 -c --perfect --num-gen $NGEN --pop-size $POPSIZE