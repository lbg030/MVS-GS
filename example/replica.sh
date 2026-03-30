#!/bin/bash

# 실행 가능한 YAML 파일 목록
CONFIG_FILES=(
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/office0.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/office1.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/office2.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/office3.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/office4.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/room0.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/room1.yaml"
  "/home/lbg030/Desktop/slam/MVS-Droid_SLAM/config/replica/room2.yaml"
)

# Python 스크립트 경로
PYTHON_SCRIPT="demo.py"

# Python 실행 경로 (환경에 따라 조정)
PYTHON_CMD="python"

# 각 config 파일로 Python 스크립트 실행
for CONFIG in "${CONFIG_FILES[@]}"; do
  echo "Running $PYTHON_SCRIPT with config $CONFIG"
  $PYTHON_CMD $PYTHON_SCRIPT --config "$CONFIG"
  
  if [ $? -ne 0 ]; then
    echo "Error occurred while running $CONFIG. Exiting."
    exit 1
  fi
done

echo "All configurations executed successfully."