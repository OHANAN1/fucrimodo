echo "Generating target file..."
fucrimodo utils \
	-a atoms_path=data/raw/test-target.xyz \
	-a save_path=data/raw/test-target.json

echo "Performing run..."
fucrimodo run \
	-c configs/run/test_run_config.py \
	-s data/results/ \
	-n test_run \
	data/raw/test-target.json

echo "Analysing run data..."
fucrimodo analyse run -r 0 data/results/test_run/

echo "Analysing stages..."
fucrimodo analyse stage -r 0 data/results/test_run/stage_1/
