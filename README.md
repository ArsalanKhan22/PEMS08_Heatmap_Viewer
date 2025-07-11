# 🚦 PEMS08 Heatmap Viewer

A Streamlit-based interactive web application for visualizing traffic flow data from the [PEMS08 dataset](https://github.com/GestaltCogTeam/BasicTS). This tool allows you to dynamically select time steps and sensor ranges, explore traffic behavior using vivid heatmaps, adjust color scales, and export high-resolution plots as PNG images.

---

## 📊 Features

- 📁 Upload and visualize `data.dat` from PEMS08 ([17856, 170, 3], float32)
- 🔍 Slice data interactively:
  - **Time steps** and **Sensor indices**
  - Select feature: `Traffic Flow`, `Occupancy`, or `Speed`
- 🎨 Adjustable **color range sliders** (vmin / vmax)
- 🖼️ Heatmap rendering with Matplotlib + Seaborn
- 💾 **Download heatmap** as PNG with a single click
- ✅ Clean, class-based modular architecture with docstrings

---

## 📂 File Structure

```bash
.
├── pems08_heatmap_viewer.py   # Main Streamlit app (class-based)
└── data.dat                   # This is Dataset file
└── README.md                  # This is documentation file

