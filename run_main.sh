#!/bin/bash
pathDatasetEuroc='/home/yongqi/workspace/58_Dataset_for_ORB_SLAM3/EuRoc_Dataset' #Example, it is necesary to change it by the dataset path

#------------------------------------
# Monocular Examples
echo "Launching MH01 with Monocular sensor"

./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml "$pathDatasetEuroc"/MH01 ./Examples/Monocular/EuRoC_TimeStamps/MH01.txt dataset-MH01_mono

#./Examples/Monocular/mono_euroc ./Vocabulary/ORBvoc.txt ./Examples/Monocular/EuRoC.yaml /home/yongqi/workspace/58_Dataset_for_ORB_SLAM3/EuRoc_Dataset/MH01 ./Examples/Monocular/EuRoC_TimeStamps/MH01.txt 


#./Examples/Stereo/stereo_euroc ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml /home/yongqi/workspace/58_Dataset_for_ORB_SLAM3/EuRoc_Dataset/MH01 ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt

