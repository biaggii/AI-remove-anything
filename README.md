# AI remover
Remove the background or anything from a video or picture.


## Installation
### python 3.10
```bash
pip install -r .\requirements.txt --ignore-installed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

## Usage
```bash
cd webui
python app.py
```

## options
```bash
--device default=None {cuda,cpu}
--sam_model_type , default='vit_h' {vit_h, vit_b}
--port default=7860
--mask_save default=False
--checkpoint_folder default='./ProPainter-Webui/ProPainter/weights'
--debug default=True
```

## TODO
- [ ] Add more models
- [ ] Add batch features
- [ ] Add different mask