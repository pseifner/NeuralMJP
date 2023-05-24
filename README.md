# NeuralMJP

Code accompanying the paper *"Neural Markov Jump Processes"* published at ICML 2023.


### Quicklinks


[NeuralMJP training loop]()

[Subblocks](), including a solver for the master equation and various prior / posterior processes

Hyperparameters are specified in [configuration files]()


### Setup

Create and activate conda environment: 
    
    conda env create --file environment.yaml
    conda activate neuralmjp

Install pytorch for your system. 

Install this package: 

    pip install -e src/


### Data Generation

Synthetic data for the Lotka-Volterra Process, the Discrete Flashing Ratchet Process and the Brownian Dynamics can be generated with:

    python3 scripts/data_generation/generate_data.py -c configs/data/lv.yaml

    python3 scripts/data_generation/generate_data.py -c configs/data/dfr.yaml

    python3 scripts/data_generation/generate_data.py -c configs/data/bd.yaml

By default, the data is saved in `data/`.


### Training 

Train NeuralMJP on the synthetic data from above data with:

    python3 scripts/train_model.py -c configs/models/lotka_volterra.yaml

    python3 scripts/train_model.py -c configs/models/flashing_ratchet.yaml

    python3 scripts/train_model.py -c configs/models/brownian_dynamics.yaml 

You can follow the training process on tensorboard: 

    tensorboard --logdir results


## Citation

If you found this code useful in your academic research, please cite: 

> @misc{}
>
>
>
