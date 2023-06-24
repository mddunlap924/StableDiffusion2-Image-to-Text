python3 -m training.main \
    --dataset-type "csv" \
    --train-data "./kaggle/diffusiondb_ds2_ds3-2000_Train_TRUNNone_filts_COCA.csv" \
    --warmup 10000 \
    --batch-size 16 \
    --lr 1e-05 \
    --wd 0.1 \
    --epochs 1 \
    --workers 8 \
    --model coca_ViT-L-14 \
    --pretrained "./kaggle/input/open-clip-models/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k.bin" \
    --report-to "wandb" \
    --wandb-project-name "StableDiffusion" \
    --coca-contrastive-loss-weight 0 \
    --coca-caption-loss-weight 1 \
    --log-every-n-steps 1000 \
    --seed 42 \
