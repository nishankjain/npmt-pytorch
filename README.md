# npmt-pytorch

# fairseq-preprocess
```
fairseq-preprocess --source-lang de --target-lang en --trainpref iwslt16.tokenized.de-en/train --validpref iwslt16.tokenized.de-en/valid --testpref iwslt16.tokenized.de-en/test --destdir data-bin/iwslt16.tokenized.de-en
```

# fairseq-train
```
fairseq-train -a npmt_iwslt_de_en --user-dir ./ data-bin/iwslt16.tokenized.de-en/ --task load_dataset --batch-size 32 --max-epoch 40 --lr 0.001 --optimizer adam --clip-norm 10
```

# fairseq-generate
```
fairseq-generate --user-dir ./ data-bin/iwslt16.tokenized.de-en/ --path checkpoints/checkpoint_best.pt --beam 10 --remove-bpe --sacrebleu | tee gen.txt
```
