Neural CEDF for Continuum Robot Control [[Paper]](https://arxiv.org)
===========================================


![4-link](result_videos_cluttered/env1/mppi/link4.gif) ![5-link](result_videos_cluttered/env1/mppi/link5.gif) ![7-link](result_videos_cluttered/env1/mppi/link7.gif)

![4-link](result_videos_sphere/env2/mppi/link4.gif) ![4-link](result_videos_sphere/env3/mppi/link4.gif) ![charger](result_videos_charging/env1/mppi/link4.gif)

This repository contains the official implementation for the paper "Neural Configuration Distance Function for Continuum Robot Control".

If you find our work useful, please consider citing our paper:
```
@article{}
```


# 🚀 Quick Start
Clone the repository: 

```
git clone 
cd 
```

## 📦 Dependencies
This code has been tested on Ubuntu 22.04 LTS. To set up the environment:

```
conda create -n environment.yaml
conda activate soft_neural_cedf
```

Note: The MPPI algorithm is computationally expensive. For real-time performance, we recommend using an NVIDIA RTX 3090 GPU or better. If no GPU is available, uncomment the following line in the relevant scripts:

jax.config.update('jax_platform_name', 'cpu')


## 🧠 Neural CEDF Training

To train the neural CEDF, run the file:
```
main_cedf.py
```

*   Adjust training parameters in training/config_3D.py
*   Default training dataset is in training_data/
*   To customize the continuum robot link size, modify robot_config.py (e.g., LINK_RADIUS, LINK_LENGTH), and run the file:

```
data_prepare_3D_link.py
```
    
to prepare the dataset for the customized continuum robot link. 



## 🤖 Continuum Robot Navigation Simulation

Customize simulation settings in robot_config.py: 
*  Number of robot links
*  Environment details (obstacles, number of simulated environments, ...)


The default N-CEDF model for navigation simualtion is trained_models/torch_models_3d/eikonal_4_16.pth


1. Dynamic Environment with Multiple Spheres, run the file
```
main_control_sphere.py
```

2. Cluttered Environment: 
```
main_control_cluttered.py
```

3. Charging Port Application: 
```
main_control_charging.py
```

Add --no_interactive flag to disable the interactive window:
```
python main_control_sphere.py --no_interactive
```

## 📊 Results

Simulation videos: results_videos_sphere/, results_videos_cluttered/ , and results_videos_charging/ . 

Distance plots: distance_plots/

