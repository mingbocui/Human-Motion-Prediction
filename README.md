## Setup
All code was developed and tested on Ubuntu 16.04 with Python 3.5 and PyTorch 0.4.

You can setup a virtual environment to run the code like this:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
echo $PWD > env/lib/python3.5/site-packages/sgan.pth  # Add current directory to python path
# Work for a while ...
deactivate  # Exit virtual environment
```

## Generate synthetic data
Go to the folder of generatedSyntheticTraj, generateTraj.py will generate sysnthetic trajectories with different number and different distribution. After generate the data, you could use the corresponsing .sh file move these trajectory to the corresponding dataset folder:\
for example: bash move3Traj.sh

## Train a model
just run the following command you will get the results of corresponding model structure, and the images are store under the images/ folder 

$python scripts/cgs_train.py --best_k 1 \
--l2_loss_weight 0 \
--Encoder_type 'MLP' \
--Decoder_type 'MLP' \
--mlp_encoder_layers 3 \
--mlp_decoder_layers 4 \
--mlp_discriminator_layers 6 \
--dataset_name 'three_traj'

by running above commands, you will get images stored under the images/ folder with folder name like \
three_traj_EN_MLP(3)_DE_MLP(4)_DIS_FF(6)_L2_Weight(0.0) 

three_traj represents we test the modle on trajectories with 3 modes; \
EN_MLP(3) represents we deploy 3 layers MLP for Encoder; \
DE_MLP(4) represents we deploy 4 layers MLP for Decoder; \
DIS_FF(6) represents we deploy 6 layers Feedforward network for Discriminator; \
L2_Weight(0.0) represents the ratio of L2 loss in our objective loss function is 0, namely we just use adversarial loss 


## Command Options

- `--best_k`: here we leave out the variety loss and set it to 1
- `--l2_loss_weight`: weight of l2 loss in the loss function, here we did not involve any ratio of l2 loss
- `--mlp_encoder_layers`: #layers of MLP structure in Encoder
- `--mlp_decoder_layers`: #layers of MLP structure in Decoder
- `--mlp_discriminator_layers`: #layers in Feedforward neural network
- `--dataset_name`: define the dataset name, for example, 'three_traj' represent the synthetic trajectories with 3 modes






