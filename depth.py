import math
import numpy as np

#獲取相機參數
# Parameters from the image
sensor_width_px = 1280  # Horizontal resolution in pixels
horizontal_fov_deg = 91  # Horizontal field of view in degrees

# Calculating the focal length # 焦距，單位毫米
focal_length = sensor_width_px / (2 * math.tan(math.radians(horizontal_fov_deg / 2)))

# Assuming sensor width is 6.4 mm as a common value for similar depth sensors
sensor_width_mm = 6.4

# 像素尺寸，單位毫米/像素 (這裡假設0.01毫米/像素)
pixel_size = sensor_width_mm / sensor_width_px

#計算長度
# 假設的深度圖和 bounding box 座標
depth_map = np.array(...)  # 你的深度圖數據
x_min, y_min, x_max, y_max = 100, 150, 200, 300  # 檢測框的座標

# 從深度圖中提取檢測框內的深度值
bounding_box_depth = depth_map[y_min:y_max, x_min:x_max]
average_depth = np.mean(bounding_box_depth)

# 計算物體的寬度和高度（物理尺寸）
bounding_box_width_px = x_max - x_min
bounding_box_height_px = y_max - y_min

# 使用公式計算實際寬度和高度
width_real = (bounding_box_width_px * average_depth / focal_length) * pixel_size
height_real = (bounding_box_height_px * average_depth / focal_length) * pixel_size

print(f"物體的實際寬度: {width_real} mm")
print(f"物體的實際高度: {height_real} mm")

