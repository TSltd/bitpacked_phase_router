rm -rf build/
rm -f router*.so
rm -f *.so

python setup.py build_ext --inplace

python evaluation/compare_original_vs_interval.py --tag original
python evaluation/compare_original_vs_interval.py --tag interval

python evaluation/compare_original_vs_interval.py --compare

python evaluation/fit_dispatch_weights.py

./scripts/run_full_pipeline.sh
