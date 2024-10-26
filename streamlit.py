import streamlit as st
from PIL import Image
import numpy as np
import cv2
from paddlex import create_model

# 设置页面的布局和标题
st.set_page_config(layout="centered", page_title="图像推理程序")

# 定义模型加载函数
@st.cache_resource
def load_model(model_path):
    print(model_path)
    return create_model(model_path, "gpu:0")

# 定义模型路径
model_paths = {
    "苹果": r"D:\project\test\apple_inference",
    "猕猴桃": r"D:\project\test\kiwi_inference"
}

# 使用侧边栏选择页面
page = st.sidebar.selectbox("选择页面", ["苹果", "猕猴桃"])

# 动态加载模型
model_path = model_paths[page]
model = load_model(model_path)

# 主体检测模型路径
object_detection_model_path = "PP-ShiTuV2_det"
object_detection_model = load_model(object_detection_model_path)

# 页面布局
st.write(f"## {page} 图像推理程序")
st.write(":dog:输入一张图像，返回预测的结果:grin:")

# 上传图片组件放在顶部
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# 开关：是否先进行主体检测
use_object_detection = st.sidebar.checkbox("先进行主体检测")

# Process and display the uploaded image
def fix_image(upload, model, use_detection):
    # Load image using PIL and convert to NumPy array
    image_pil = Image.open(upload)
    image_np = np.array(image_pil)

    # Convert RGB to BGR for OpenCV
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.write("### Image :camera:")
    st.image(image_pil, use_column_width=True)

    if use_detection:
        # 先进行主体检测
        print("主体检测")
        detection_output = object_detection_model.predict(image_cv2)
        
        for res in detection_output:
             # 检查是否有检测到目标
            if len(res["boxes"]) > 0:
                for j, box in enumerate(res["boxes"]):
                    coordinates = box["coordinate"]
                    res.print(json_format=False)
                    x_min, y_min, x_max, y_max = list(map(int, coordinates))
                    cropped_img = image_cv2[y_min:y_max, x_min:x_max]
                    # 调用分类模型
                    print("分类")
                    output = model.predict(cropped_img, device="gpu:0", batch_size=1)
                    for res in output:
                        res.print(json_format=False)
                        st.write(f"Label: {res['label_names'][0]}")
                        st.write(f"Score: {res['scores'][0]}")
                        # 绘制检测框
                        cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            else:
                print(f"No objects detected in image")
            
    else:
        # 直接调用分类模型
        print("分类2")
        output = model.predict(image_cv2)
        for res in output:
            res.print(json_format=False)
            st.write(f"Label: {res['label_names'][0]}")
            st.write(f"Score: {res['scores'][0]}")

    # 显示最终结果图像
    st.write("### Result Image :camera:")
    result_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    st.image(result_image, use_column_width=True)

# 上传图片组件
my_upload = st.file_uploader("上传图像", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("上传的文件太大，请上传小于10MB的图像。")
    else:
        fix_image(upload=my_upload, model=model, use_detection=use_object_detection)
