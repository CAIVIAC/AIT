while getopts b:g:w: option
do
    case "${option}"
        in
        b) BATCH_SIZE=${OPTARG};;
        g) GPUS=${OPTARG};;
        w) WORKER_NUMBER=${OPTARG};;
        #p) PRODUCT=${OPTARG};;
        #f) FORMAT=${OPTARG};;
    esac
done

BATCH_SIZE=8
WORKER_NUMBER=8
LEARNING_RATE=0.01
DECAY_STEP=4
SPLIT=0
SEEN=1
SESSION=1
VERSION=0.0.0
GPUS=4
EPOCH=10
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net_voc.py\
    --dataset pascal_voc_0712 --net res50 \
    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
    --cuda --g $SPLIT --seen $SEEN --session $SESSION \
    --version $VERSION \
    --mGPUs --gpus $GPUS --epochs $EPOCH
