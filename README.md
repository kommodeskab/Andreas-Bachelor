### How to run an experiment
To run an experiment, launch it from the console. For example, to run the ```fr_schrodinger.yaml``` experiment (found in configs/experiment), simply write:
```
python train.py +experiment=mnist_to_emnist trainer.max_epochs=10
```
To run the same experiment with different parameters, for example with a different batch size, either edit the config files or manually update directly in the console:
```
python train.py +experiment=mnist_to_emnist trainer.max_epochs=10 data.batch_size=8 
```
