# DeliSense: Joint Food Delivery and Urban Sensing with Human-Robot Collaboration

### Experiment Data Files

- [ ] Experiment Order data (Eleme): **data/order_data_test_2020_1013-1014_zhonghuan.csv** .


### Code Files
- [ ] **envs.py**: Contains the environment class, sensing implementations, and utility functions to update the environment after order dispatch and sensing execution.

- [ ] **ksubmod.py**: Contains the MapReduce K-Submodular order dispatch module.

- [ ] **baselines.py**: Contains all baselines.

### How run
-  Run Delisense:
    -  Run mpirun -np **#Processes** --bind-to none python3 -W ignore *ksubmod.py* **#AVS**.
-  Run Fastest-Delivery (FastD):
    -  Run *python *baselines.py* **fdd-sdp** **#AVS**.
-  Run Best-Sensing (BestS):
    -  Run *python *baselines.py* **bsd-sdp** **no. of AVS** *.
-  Run Joint-DeliverySensing-Efficiency (JointDS):
    -  Run *python *baselines.py* **dsd-rsn** **no. of AVS** *.