# Learning from Label Proportions with Instance-wise Consistency

## Install
We use `python=3.10` and [poetry](https://python-poetry.org/docs/).
- Install poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```
- Install python package
```
poetry install --only main
```

## Download
```
# at src/

poetry run sh runs/preprocess.sh
```

## Training
```
# at src/

# runs/example.sh
# python main.py -cp yamls/base.yaml yamls/models/${1}.yaml yamls/data/${2}.yaml \
#  yamls/Ks/K${3}.yaml yamls/method/${4}.yaml yamls/lrs/lr1e-${5}.yaml --seed ${6}

# you can change settings using yamls at src/yamls/.

poetry run sh runs/example.sh resnet cifar10 4 rc 3 0
```

## Links
- [arXiv](https://arxiv.org/abs/2203.12836)
