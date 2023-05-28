# Speech Intelligibility Assessment of Dysarthric Speech by using Goodness of Pronunciation with Uncertainty Quantification
Official implementation of the paper.

## How to run
```
# Dataset cleansing
python3 dataset.py --output_path /path/to/output --dataset_path /path/to/dataset --dataset_type commonphone
python3 dataset.py --output_path /path/to/output --dataset_path /path/to/dataset --dataset_type l2arctic
python3 dataset.py --output_path /path/to/output --dataset_path /path/to/dataset --dataset_type ssnce
python3 dataset.py --output_path /path/to/output --dataset_path /path/to/dataset --dataset_type qolt
python3 dataset.py --output_path /path/to/output --dataset_path /path/to/dataset --dataset_type uaspeech
# You have to manually merge l2arctic & commonphone

# Train phone recognizer
python3 train_phone_recognizer.py \
    --exp_name both_phonewise_morereducevocab_average \
    --num_epochs 4 \
    --learning_rate 0.001 \
    --loss phonewise_average \
    --gpu 0 \
    --model facebook/wav2vec2-xls-r-300m \
    --use_conv_only True \
    --batch_size 4 \
    --commonphone_csv /path/to/commonphone_l2arctic.csv.gz \
    --reduce_vocab True

# Evaluate gop
for dataset in uaspeech qolt ssnce
do
    python3 gop.py \
        --dataset_csv /path/to/$dataset.csv.gz \
        --commonphone_csv /path/to/commonphone_l2arctic.csv.gz \
        --gpu 0 \
        --model_path /path/to/model
done
```

## Full phone Kendall's tau table per langauge
```
- English
    aɪ -0.5363
    ʃ -0.5036
    aʊ -0.4073
    z -0.4056
    dʒ -0.3922
    æ -0.3857
    a -0.3594
    o -0.3517
    s -0.3433
    tʃ -0.3233
    e -0.2907
    t -0.2812
    h -0.2704
    ɔ -0.2659
    ɛ -0.2489
    r -0.2462
    j -0.2345
    f -0.2072
    əʊ -0.1972
    ɔɪ -0.1879
    ʌ -0.1776
    d -0.1748
    k -0.1676
    ʎ -0.1605
    p -0.1526
    ʔ -0.1503
    v -0.1395
    u 0.1199
    ɡ -0.1075
    i -0.1041
    w -0.1007
    ŋ -0.0973
    n -0.0958
    θ -0.0794
    ʊ -0.0723
    m -0.0458
    b -0.0233
    l -0.0155
    ð -0.0097
    
- Korean
    i -0.2595
    s -0.2383
    n -0.2272
    a -0.2080
    ʌ -0.1983
    l -0.1849
    e -0.1792
    ɨ 0.1682
    d -0.1633
    u -0.1406
    o -0.1315
    w -0.1291
    t -0.1008
    j 0.0843
    ɡ 0.0745
    ts 0.0733
    m -0.0688
    ɛ -0.0640
    h 0.0494
    k -0.0470
    p -0.0437
    r -0.0306
    ŋ -0.0252
    
- Tamil
    ʃ -0.3013
    h -0.2778
    tʃ -0.2410
    dʒ -0.2066
    aɪ -0.1868
    ɣ -0.1858
    s -0.1840
    m -0.1702
    u -0.1687
    i -0.1676
    b -0.1670
    l -0.1667
    n -0.1620
    d -0.1538
    ŋ -0.1524
    o -0.1500
    k -0.1449
    ð -0.1337
    r -0.1311
    p -0.1189
    w -0.1175
    a -0.1069
    ɨ -0.1031
    θ -0.1006
    ɡ -0.0949
    e -0.0907
    t -0.0405
```
