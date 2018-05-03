export PYTHONPATH=./:$PYTHONPATH
../caffe/build/tools/caffe train --solver=solver.prototxt 2>&1 | tee log_`date +%Y-%m-%d-%H-%M-%S`.log
