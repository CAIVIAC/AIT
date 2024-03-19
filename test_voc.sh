DEVICE=0
CHECKEPOCH=10
SESSION=1
CHECKPOINT=3514
SPLIT=0
AVERAGE=4
VERSION=0.0.0

CUDA_VISIBLE_DEVICES=$DEVICE python test_net_voc.py \
    --dataset pascal_voc_0712 --net res50 \
    --s $SESSION --checkepoch $CHECKEPOCH \
    --p $CHECKPOINT --cuda --g $SPLIT \
    --a $AVERAGE --version $VERSION
