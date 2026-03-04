# 🚀 HIAS-KCF Optical Tracker (高精度光电追踪系统)

## 📌 项目概述
结合深度学习（YOLOv11）与运动估计（Kalman Filter）的工业级视觉遥测系统。

## 🎯 核心技术
1. **域适应训练**: 解决廉价CMOS噪点畸变，mAP0.5-0.95达 0.951。
2. **KCF 状态估计**: 引入卡尔曼滤波，实现抗遮挡惯性跟踪（黄色预测环）。
3. **硬件在环(HIL)**: 包含6字节定长通信协议与HUD动态波形遥测。

## 🚀 快速启动
1. 安装依赖: `pip install -r requirements.txt`
2. 将训练好的模型权重改名为 `final_model.pt` 并放入 `weights/` 目录。
3. 检查 `config.py` 中的串口和摄像头配置。
4. 运行: `python main.py`

## 👨‍💻 作者
李冠儒 | 中国计量大学
