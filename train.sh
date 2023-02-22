python3 train_phone_recognizer.py \
    --exp_name baseline \
    --num_epochs 1 \
    --loss ctc_like \
    --model facebook/wav2vec2-xls-r-300m \
    --use_conv_only True \
    --commonphone_csv /home/eunjung/workspace/interspeech2023/dysarthria-gop/commonphone.csv.gz
