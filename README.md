# MIDI-Music-Classification-Genre-and-Mood-using-REMI-z


## Prepare Environment
```
# Environment
conda create -n dev python=3.8
conda activate dev

# Install PyTorch
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install -r requirements.txt   

git clone https://github.com/Sonata165/REMI-z.git
cd REMI-z
pip install -r Requirements.txt
pip install -e .
```
