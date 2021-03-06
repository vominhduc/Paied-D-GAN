. CONFIG

python train.py \
    --img_root /PATH/TO/IMAGE/ROOT \
    --caption_root PATH/TO/IMAGE/CAPTION/ROOT \
    --trainclasses_file trainvalclasses.txt \
    --fasttext_model /PATH/TO/FASTTEXTMODEL \
    --text_embedding_model ./models/text_embedding_birds.pth \
    --save_filename ./models/birds.pth