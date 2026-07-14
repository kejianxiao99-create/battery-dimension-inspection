from pathlib import Path

import cv2


def main() -> None:
    # 1. 计算项目根目录
    project_root = Path(__file__).resolve().parents[1]

    # 2. 设置输入和输出路径
    image_path = project_root / "data" / "raw" / "battery_01.jpg"
    result_path = project_root / "results" / "battery_detection.jpg"

    # 3. 读取图片
    image = cv2.imread(str(image_path))

    # 4. 检查读取是否成功
    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # 5. 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 6. 高斯模糊，降低噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 7. 自动二值化
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # 8. 查找外部轮廓
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # 9. 检查是否找到轮廓
    if not contours:
        raise RuntimeError("No object contour was detected.")

    # 10. 选择面积最大的轮廓，暂时认为它是电池
    battery_contour = max(contours, key=cv2.contourArea)

    # 11. 计算外接矩形
    x, y, width, height = cv2.boundingRect(battery_contour)

    # 12. 复制原图，用于绘制结果
    output = image.copy()

    # 13. 绘制矩形框
    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2,
    )

    # 14. 添加尺寸文字
    label = f"Width: {width}px, Height: {height}px"

    cv2.putText(
        output,
        label,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 15. 创建结果文件夹
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # 16. 保存结果图片
    success = cv2.imwrite(str(result_path), output)

    if not success:
        raise RuntimeError(f"Could not save result: {result_path}")

    # 17. 在控制台输出结果
    print(f"Image path: {image_path}")
    print(f"Battery width: {width} pixels")
    print(f"Battery height: {height} pixels")
    print(f"Result saved to: {result_path}")

    # 18. 显示中间过程和最终结果
    cv2.imshow("Original Image", image)
    cv2.imshow("Binary Image", binary)
    cv2.imshow("Detection Result", output)

    # 19. 等待按键
    cv2.waitKey(0)

    # 20. 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()