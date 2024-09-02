from ultralytics import YOLO
import cv2
import torch
import math
import numpy as np
rgb=cv2.imread("./Depth2/000890_0000_237040_1723358708980_Color_1280x720.png")
depth=cv2.imread("./Depth2/000927_0000_237020_1723358708961_Depth_1280x800.png",0)


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

h1,w1=rgb.shape[0:2]
h2,w2=depth.shape[0:2]
diff=h2-h1
padding=int(diff/2)
depth=depth[padding:padding+h1,:]


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=YOLO("./yolov8n.pt").to(device)
# Predict with the model
results = model(rgb, stream=True)


for result in results:
    boxes = result.boxes
    for box in boxes:
        score=box.conf[0].cpu().numpy()
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy()
        x1, y1, x2, y2 =int(x1), int(y1), int(x2), int(y2)
        xc,yc=int((x1+x2)/2),int((y1+y2)/2)

        cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,0,255),1)
        # 從深度圖中提取檢測框內的深度值
        bounding_box_depth = depth[y1:y2, x1:x2]*100
        average_depth = np.mean(bounding_box_depth)

        # 計算物體的寬度和高度（物理尺寸）
        bounding_box_width_px = x2 - x1
        bounding_box_height_px = y2 - y1
        print("pixel:",bounding_box_width_px,bounding_box_height_px)

        # 使用公式計算實際寬度和高度
        width_real = (bounding_box_width_px * average_depth / focal_length) * pixel_size
        height_real = (bounding_box_height_px * average_depth / focal_length) * pixel_size

        text= str(width_real)+"x"+str(height_real)
        cv2.putText(rgb,text,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

cv2.imwrite("result.jpg",rgb)
