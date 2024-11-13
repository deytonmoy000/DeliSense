HRSense: Joint Urban Sensing and Crowdsourcing with Human-Robot Collaboration
### Experiment Data Files

- [ ] Experiment Order data (Eleme): **data/order_data_test_2020_1013-1014_zhonghuan.csv** .


### Code Files
- [ ] **envs.py**: Contains the environment class, sensing implementations, and utility functions to update the environment after order dispatch and sensing execution.

- [ ] **ksubmod.py**: Contains the MapReduce K-Submodular order dispatch module.

- [ ] **baselines.py**: Contains all baselines.

### How run
-  Run HRsense:
    -  Run mpirun -np **#Processes** --bind-to none python3 -W ignore *ksubmod.py* **#AVS** **r^T coefficient** **r^C coefficient** **r^I coefficient** deliSense
-  Run Fastest-Delivery (FastD):
    -  Run *python *baselines.py* **fdd-sdp** **#AVS**.
-  Run Best-Sensing (BestS):
    -  Run *python *baselines.py* **bsd-sdp** **no. of AVS** *.
-  Run Joint-DeliverySensing-Efficiency (JointDS):
    -  Run *python *baselines.py* **dsd-rsn** **no. of AVS** *.
-  Run LSTAlloc :
    -  Run *python *baselines.py* **lsta-sdp** **no. of AVS** *.
-  Run AJRP :
    -  Run *python *baselines.py* **ajrp-sdp** **no. of AVS** *.    