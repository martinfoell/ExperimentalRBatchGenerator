n_trainings=$1
int_n_trainings=$((n_trainings))

python3 ../python/multi-physics-bm-Higgs-NN.py 200000 100000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 200000 50000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 200000 25000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 200000 8000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 200000 4000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 200000 2000 $int_n_trainings

python3 ../python/multi-physics-bm-Higgs-NN.py 100000 50000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 100000 25000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 100000 12500 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 100000 4000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 100000 2000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 100000 1000 $int_n_trainings

python3 ../python/multi-physics-bm-Higgs-NN.py 50000 25000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 50000 12500 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 50000 6250 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 50000 2000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 50000 1000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 50000 500 $int_n_trainings

python3 ../python/multi-physics-bm-Higgs-NN.py 25000 12500 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 25000 6250 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 25000 3125 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 25000 1000 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 25000 500 $int_n_trainings
python3 ../python/multi-physics-bm-Higgs-NN.py 25000 250 $int_n_trainings

python3 ../python/merge_NN_output.py

root ../src/plot_auc-diff_signal_frac.C



# plots saved in the plots/ folder
