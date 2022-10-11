# Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation
This repository contains the official implementation of our NeurIPS 2022 paper [Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation](https://arxiv.org/pdf/2210.03953.pdf). Our code is implemented based on the open-source toolkit [fairseq](https://github.com/pytorch/fairseq). We implement our training objectives in [nat_loss.py](https://github.com/ictnlp/NMLA-NAT/blob/master/fairseq/criterions/nat_loss.py).

# Requirements
This system has been tested in the following environment.

+ Python version = 3.8
+ Pytorch version = 1.7

# Knowledge Distillation
Knowledge distillation from an autoregressive model can effectively simplify the training data distribution. You can directly download the [distillation dataset](http://dl.fbaipublicfiles.com/nat/distill_dataset.zip), or you can follow the instructions of [training a standard transformer model](https://github.com/facebookresearch/fairseq/tree/main/examples/translation), and then decode the training set to produce a distillation dataset for NAT. 

# Preprocess
We provide the scripts for replicating the results on WMT14 En-De. For other tasks, you need to adapt some hyperparameters accordingly. First, preprocess the distillation dataset.
```bash
TEXT=your_data_dir
python preprocess.py --source-lang en --target-lang de \
   --trainpref $TEXT/train.en-de --validpref $TEXT/valid.en-de --testpref $TEXT/test.en-de \
   --destdir data-bin/wmt14ende_dis --workers 32 --joined-dictionary
```

# Pretrain
Train a CTC model on the distillation dataset.
```bash
data_dir=data-bin/wmt14ende_dis
save_dir=output/wmt14ende_ctc
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --use-word --src-embedding-copy --fp16 --ddp-backend=no_c10d --save-dir $save_dir \
    --task translation_lev \
    --criterion ctc_loss \
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)'  \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' \
    --dropout 0.2 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --max-tokens 4096 --update-freq 2\
    --save-interval-updates 5000 \
    --max-update 300000 --keep-interval-updates 5 --keep-last-epochs 5

sh average.sh $save_dir
```

# Finetune
Finetune the CTC model with the NMLA objective.
```bash
model_dir=output/wmt14ende_ctc
data_dir=data-bin/wmt14ende_dis
save_dir=output/wmt14ende_nmla
mkdir $save_dir
cp $model_dir/average-model.pt $save_dir/checkpoint_last.pt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --reset-optimizer --src-embedding-copy --fp16  --use-ngram --ngram-size 2 --ddp-backend=no_c10d --save-dir $save_dir \
    --task translation_lev \
    --criterion ctc_loss \
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)'  \
    --lr 0.0003 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 1 \
    --max-tokens 2048 --update-freq 16\
    --save-interval-updates 500 \
    --max-update 6000

sh average.sh $save_dir
```

# Deocde
We can decode the test set with argmax decoding:
```bash
data_dir=data-bin/wmt14ende_dis
model_dir=output/wmt14ende_nmla/average-model.pt
CUDA_VISIBLE_DEVICES=0 python generate.py $data_dir \
    --gen-subset test \
    --task translation_lev \
    --iter-decode-max-iter  0  \
    --iter-decode-eos-penalty 0 \
    --path $model_dir \
    --beam 1  \
    --left-pad-source False \
    --batch-size 100 > out
# because fairseq's output is unordered, we need to recover its order
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.raw
python collapse.py
sed -r 's/(@@ )|(@@ ?$)//g' pred.de.collapse > pred.de
perl multi-bleu.perl ref.de < pred.de
```

We can also apply beam search decoding combined with a 4-gram language model to search the target sentence. First, install the ctcdecode package.
```bash
git clone --recursive https://github.com/MultiPath/ctcdecode.git
cd ctcdecode && pip install .
```
Notice that it is important to install [MultiPath/ctcdecode](https://github.com/MultiPath/ctcdecode) rather than the original package. This version pre-computes the top-K candidates before running the beam-search, which makes the decoding much faster. Then, follow [kenlm](https://github.com/kpu/kenlm) to train a target-side 4-gram language model and save it as ``wmt14ende.arpa``. Finally, decode the test set with beam search decoding combined with a 4-gram language model.
```bash
data_dir=data-bin/wmt14ende_dis
model_dir=output/wmt14ende_nmla/average-model.pt
CUDA_VISIBLE_DEVICES=0 python generate.py $data_dir \
    --use-beamlm \
    --beamlm-path ./wmt14ende.arpa \
    --alpha $1 \
    --beta $2 \
    --gen-subset test \
    --task translation_lev \
    --iter-decode-max-iter  0  \
    --iter-decode-eos-penalty 0 \
    --path $model_dir \
    --beam 1  \
    --left-pad-source False \
    --batch-size 100 > out
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.raw
sed -r 's/(@@ )|(@@ ?$)//g' pred.raw > pred.de
perl multi-bleu.perl ref.de < pred.de
```
The optimal choices of alpha and beta vary among datasets and can be found by grid-search.

# Other Models

To implement DDRS+NMLA, please follow the guidline in [DDRS-NAT](https://github.com/ictnlp/DDRS-NAT), where we have supported the NMLA objective there. It is also convenient to implement NMLA on other CTC-Based models, where you only need to copy the compute_ctc_bigram_loss function in [nat_loss.py](https://github.com/ictnlp/NMLA-NAT/blob/master/fairseq/criterions/nat_loss.py) and paste it to your loss file.

To implement SCTC, you need to replace the pytorch source file pytorch/aten/src/ATen/native/cuda/LossCTC.cu with our file [LossCTC.cu](https://github.com/ictnlp/NMLA-NAT/blob/master/LossCTC.cu) and then recompile pytorch. After recompilation, the built-in function F.ctc_loss will become SCTC.

# Citation 
If you find the resources in this repository useful, please cite as:

``` bibtex
@inproceedings{nmla,
  title = {Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation},
  author= {Chenze Shao and Yang Feng},
  booktitle = {Proceedings of NeurIPS 2022},
  year = {2022},
}
```
