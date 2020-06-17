# Semantics-AssistedVideoCaptioning.Pytorch
This is unofficial implementation of [Semantics-Assisted Video Captioning Model Trained with Scheduled Sampling Strategy](https://arxiv.org/abs/1909.00121).
You can find official tensorflow implementation [here](https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning).

## Dependency
* python 3.6
* pytorch
```shell script
pip install -r requirements.txt
```
## Steps
### Data
Please download data in official [repo](https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning) and put them
all in ./data
### Training
If you want to train on msvd with default parameters:
```shell script
python train.py --cfg configs/msvd_default.yml --savedir saved_results --exp_name anonymous_run
```
Remember to create savedir by `mkdir`. `--exp_name` is the name you give to this run.

For training on msrvtt, just change `--cfg` to `configs/msrvtt_default.yml`. It
takes about 90 min to train on msvd, 5h to train on msr-vtt (on GTX 1080Ti).

For more details about configs, please see `opts.py` and yaml files in ./configs
### Babysitting
You can see training process by tensorboard.
```shell script
tensorboard --logdir saved_results --port my_port --host 0.0.0.0
```
### Evaluation
```shell script
python eval.py --savedir saved_results --exp_name anonymous_run --max_sent_len 20 --model_path path_of_model_to_eval
```
If you don't specify `--model_path`, best model will be evaluated.
## Results
Results of my implementation are not chosen. I just run once for each dataset.
My implementation is comparable to official claim.
### MSVD
|Model|B-4|R|M|C|
|---|---|---|---|---|
|official|61.8|76.8|37.8|103.0|
|mine|61.2|76.6|38.5|106.5|

### MSR-VTT
|Model|B-4|C|M|R|
|---|---|---|---|---|
|official|43.8|62.4|28.9|51.4|
|mine|44.4|62.7|28.8|50.7|

## Differences
### Adam optimizer
Since Tensorflow and Pytorch implement Adam differently. I also offer
tensorflow version of Adam in optim.py. But I found they perform
comparable. So I choose Pytorch Adam by default. See more detials in
reference.
### model choice
Official implementation choose best model by a weighted sum of all scores.
I just choose model of best cider on validation set.
### dropout position
Official implementation do dropout after schedule sampling. I do it before.

## TODO(or Neverdo)
* beam search
* reinforcement learning

## Acknowledgement
Thanks the original tensorflow implementation.

## References
* adam problem 
    * https://discuss.pytorch.org/t/pytorch-adam-vs-tensorflow-adam/74471
    * https://stackoverflow.com/questions/57824804/epsilon-parameter-in-adam-opitmizer
    * https://github.com/tensorflow/tensorflow/issues/35102
* [official implementation](https://github.com/WingsBrokenAngel/Semantics-AssistedVideoCaptioning)