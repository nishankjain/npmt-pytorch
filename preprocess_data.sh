
#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

# Refer http://www.statmt.org/moses/?n=FactoredTraining.PrepareTraining
# for training data preparation

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWERCASE=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=10000

ZIP_TRAIN_VAL=iwslt16_en_de.zip
ZIP_TEST=iwslt16_en_de_test.zip

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
# prep=iwslt16.tokenized.de-en_tokenized
prep=iwslt16.tokenized.de-en
tmp=$prep/tmp
orig=orig

mkdir -p $tmp $prep

cd $orig

if [ -f $ZIP_TRAIN_VAL ]; then
    echo "Train and Val data present."
else
    echo "Train and Val data not present."
    exit
fi


if [ -f $ZIP_TEST ]; then
    echo "Test data present."
else
    echo "Test data not present."
    exit
fi


echo "Unzipping data..."

unzip $ZIP_TRAIN_VAL -d $lang
unzip $ZIP_TEST -d $lang
cd ..


echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.$l
    tok=train.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train_temp 1 50


echo "pre-processing dev data..."
for l in $src $tgt; do
    f=dev.$l
    # tok=valid.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $prep/$f
    echo ""
done


echo "pre-processing test data..."
for l in $src; do
    f=test.$l
    # tok=test.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $prep/$f
    echo ""
done


echo "creating train, valid data..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train_temp.$l > $prep/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train_temp.$l > $prep/train.$l
done
echo ""


# TRAIN=$tmp/train.en-de
# BPE_CODE=$prep/code
# rm -f $TRAIN
# for l in $src $tgt; do
#     cat $prep/train.$l >> $TRAIN
# done


# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# for L in $src $tgt; do
#     for f in train.$L valid.$L dev.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $final/$f
#     done
# done


# for L in $src; do
#     for f in test.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $final/$f
#     done
# done