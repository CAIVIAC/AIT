DEVICE=0
CHECKEPOCH=10
SESSION=1
CHECKPOINT=26623
SPLIT=1
AVERAGE=4
VERSION=0.0.0

CUDA_VISIBLE_DEVICES=$DEVICE python test_net_coco.py \
    --dataset coco --net res50 \
    --s $SESSION --checkepoch $CHECKEPOCH \
    --p $CHECKPOINT --cuda --g $SPLIT \
    --a $AVERAGE --version $VERSION
