NGEN=100
POPSIZE=60

python ./ga_hc.py ../datasets/polvo/full_scenario 10 -c -e --num-gen $NGEN --pop-size $POPSIZE
python ./ga_hc.py ../datasets/MargemEquatorial/full_scenario/ 20 -c -e --num-gen $NGEN --pop-size $POPSIZE

# python ./ga_hc.py ../datasets/polvo/full_scenario 10 -c --perfect --num-gen $NGEN --pop-size $POPSIZE
# python ./ga_hc.py ../datasets/MargemEquatorial/full_scenario/ 20 -c --perfect --num-gen $NGEN --pop-size $POPSIZE