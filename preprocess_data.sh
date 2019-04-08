
#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=10000

ZIP=iwslt16_en_de.zip

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt16.tokenized.de-en
tmp=$prep/tmp
orig=orig

mkdir -p $tmp $prep

cd $orig

if [ -f $ZIP ]; then
    echo "Data present."
else
    echo "Data not present."
    exit
fi

echo "Unzipping data..."

unzip $ZIP -d $lang
cd ..


echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.$l
    tok=train.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.clean.$l > $tmp/train.$l
done


echo "pre-processing dev data..."
for l in $src $tgt; do
    f=dev.$l
    tok=dev.tok.$l

    cat $orig/$lang/$f | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/dev.tok $src $tgt $tmp/dev.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/dev.clean.$l > $tmp/dev.$l
    cp -r $tmp/dev.$l $prep/dev.$l
done


# NR = Number of records in the input file
echo "creating train, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.$l > $prep/test.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.$l > $prep/train.$l
done


# TRAIN=$tmp/train.en-de
# BPE_CODE=$prep/code
# rm -f $TRAIN
# for l in $src $tgt; do
#     cat $tmp/train.$l >> $TRAIN
# done


# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# for L in $src $tgt; do
#     for f in train.$L valid.$L test.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
#     done
# done