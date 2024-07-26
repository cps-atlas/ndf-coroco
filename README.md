Neural CEDF for Continuum Robot Control
===========================================

[[Website]](https://github.com)
[[Paper]](https://arxiv.org)

This repository contains official implementations for the work "Neural Configuration Distance Function for Continuum Robot Control".

If you find our work useful, please consider citing our paper:
```
@article{}
```

# 🛝 Try it out!
```
git clone 
```

## Dependencies
The code is tested on Ubuntu 22.04 LTS. 

```
conda create -n environment.yaml
```

The MPPI algorithm is computative expensive, to allow for real-time performance, it is recommended to have a Nvidia 3090 GPU or better. Othewise the simulation may be slow. 

You can uncomment the line "jax.config.update('jax_platform_name', 'cpu')" is no GPU is enabled. 

## 🛠️ Neural CEDF Training
To reproduce the training results for the neural CEDF, run the command
```
python main_cedf.py
```

The training parameters can be tuned in the training/config_3D.py file (e.g., number of layers, number of neruons, learning rate ...)

The training data is prepared by the training_data/data_prepare_3D_Link.py file, and the training dataset is also saved under the training_data directory. 

If you would like to train the CEDF for a continuum robot link with different size (length and radius), you can modify the parameters in robot_config.py (e.g., LINK_RADIUS, LINK_LENGTH)

And run the command:
```
python data_prepare_3D_link.py
```
to prepare the data for the new continuum robot link

and train the network with the customized link parameters. 



## 🛠️ Continuum Robot Navigation Simulation

You can modify the number of links of the continuum robots in the ''robot_config.py'' file: 

You can also modify the dynamical environments for simulation in the robot_config.py file: 
e.g.: number of obstacles, obstacle radius, and also number of different random environments you would like to simualte


1. To run the simulation in the dynamic environment with multiple spheres, run the command 
```
main_control_sphere.py
```

2. To run the simulation in the cluttered environment, run the file
```
main_control_cluttered.py
```

If you want to disable the interactive window, run the files with the flag --no_interactive . 


The resulting videos will be saved in the results_videos_sphere or results_videos_cluttered folder, the distance plots to the obstalces will be saved in the distance_plots folder.

