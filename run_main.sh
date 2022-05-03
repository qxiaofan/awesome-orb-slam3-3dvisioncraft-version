#!/bin/bash

#euroc_data download for test: 

#链接: https://pan.baidu.com/s/1YEt1WNl1cHdgHWdbtWQDOQ?pwd=45uk 提取码: 45uk 

pathDatasetEuroc='YOUR_PATH/EuRoc_data' #Example, it is necesary to change it by the dataset path

# Monocular-Inertial Examples
echo "Launching MH01 with Monocular-Inertial sensor"
./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.bin ./Examples/Monocular-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Monocular-Inertial/EuRoC_TimeStamps/MH01.txt dataset-MH01_monoi


