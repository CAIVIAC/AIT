GPU_ID=0
BATCH_SIZE=16
WORKER_NUMBER=8
SPLIT=1
SEEN=1
LEARNING_RATE=0.01
DECAY_STEP=4
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
    --dataset coco --net res50 \
    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
    --cuda --g $SPLIT --seen $SEEN
