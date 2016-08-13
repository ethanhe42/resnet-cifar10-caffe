
set -x

GPUs=$1
NET=resnet-20
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

solver=resnet-20/solver.prototxt
LOG="resnet-20/logs/${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

~/caffe/build/tools/caffe train -gpu ${GPUs} \
    -solver ${solver} \
    ${EXTRA_ARGS}

set +x