Use python3.13
Make virtual environment
```
python -m venv .venv
```

After installing the requirements.txt, install pytorch
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

You may use requirements-server.txt but it is better to just let it build everything


start with collection, then view, then preprocess

Run by doing
```
python -m src.data.collection
```
```
python -m src.scripts.train_sl
```

Don't forget about the sysid_config like `include_rl_dataset` and `use_sl_with_rl_model`
