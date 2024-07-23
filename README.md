Neural CSDF for Continuum Robot Control
===========================================

[[Website]](https://github.com)
[[Paper]](https://arxiv.org)

This repository contains implementations for the work "Neural Configuration Distance Function for Continuum Robot Control".

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

## 🛠️ Simulation

You can modify the configurations in the robot_config.py file: 
e.g.: number of links, 
You can also modify the dynamical environments for simulation in the robot_config.py file: 
e.g.: number of obstacles, obstacle radius, and also number of different random environments you would like to simualte


1. To run the simulation in the dynamic environment with multiple spheres, run the file
```
main_control_sphere_interactive.py
```


2. To run the simulation in the cluttered environment, run the file
```
main_control_cluttered_interactive.py
```

The resulting videos will be saved in the results_videos_sphere or results_videos_cluttered folder. 

