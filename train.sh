
set -x

GPUs=$1
# eg. 0,1,2,3
NET=$2
mkdir ${NET}/logs
mkdir ${NET}/snapshot
# eg. resnet-20
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

solver=${NET}/solver.prototxt
LOG="${NET}/logs/${NET}_${EXTRA_ARGS_SLUG}_`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# specify your caffe path here
~/caffe/build/tools/caffe train -gpu ${GPUs} \
    -solver ${solver} \
    -sighup_effect stop \
    ${EXTRA_ARGS}

set +x
