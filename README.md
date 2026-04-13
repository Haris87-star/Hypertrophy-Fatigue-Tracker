# 🏋️‍♂️ Hypertrophy Fatigue Tracker

**Built for the AI/ML Track at NMIT HACKS 2026**

![App Demo Placeholder](https://via.placeholder.com/800x400.png?text=Upload+Your+Screenshot+Here)
*(Note: Replace the placeholder link above with a screenshot of your app running with the RED fatigue border!)*

## 📌 The Problem
Optimizing muscle growth (hypertrophy) requires athletes to train close to muscular failure. Professional athletes use expensive biometric sensors and hardware velocity trackers to monitor the micro-drops in lifting speed that signal fatigue. The everyday gym-goer does not have access to this budget-intensive hardware.

## 💡 Our Solution
The **Hypertrophy Fatigue Tracker** is a zero-budget, highly scalable AI software solution. Using a standard webcam, our application utilizes real-time computer vision and a custom mathematical state-machine to track lifting biomechanics. It dynamically calculates user baseline speeds and visually flags muscular fatigue the moment lifting velocity drops below an optimal threshold, allowing users to maximize muscle growth safely and effectively.

## ⚙️ Tech Stack
* **Language:** Python 3.12
* **Computer Vision:** Google MediaPipe (Pose Estimation) & OpenCV
* **Mathematics & Logic:** Custom algorithm tracking $\Delta Y / \Delta T$ (Velocity)
* **Data Logging:** Python `csv` module for local enterprise-ready data storage

## 🚀 How to Run Locally

**1. Clone the repository and navigate to the directory:**
```bash
git clone [https://github.com/YourUsername/Hypertrophy-Tracker.git](https://github.com/YourUsername/Hypertrophy-Tracker.git)
cd Hypertrophy-Tracker
