https://huggingface.co/Salesforce/blip-image-captioning-base#using-the-pytorch-model

*How to run:
run these commands inside project folder
python -m venv blip_env
source blip_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers Pillow requests flask google-generativeai
