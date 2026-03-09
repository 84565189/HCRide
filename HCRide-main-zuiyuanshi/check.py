import pickle
import os


def extract_pkl_to_txt(pkl_path, output_txt_path, dataset_name):
    """
    通用函数：将单个pkl文件提取并保存为可读的TXT文件
    :param pkl_path: 输入的pkl文件路径
    :param output_txt_path: 输出的TXT文件路径
    :param dataset_name: 数据集名称（用于打印提示）
    """
    # 检查pkl文件是否存在
    if not os.path.exists(pkl_path):
        print(f"❌ 错误：{dataset_name} 文件不存在！路径：{os.path.abspath(pkl_path)}")
        return

    try:
        # 读取pkl文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # 写入TXT文件
        with open(output_txt_path, "w", encoding="utf-8") as f:
            # 1. 写入基础信息
            f.write(f"📊 数据集名称：{dataset_name}\n")
            f.write(f"📏 数据类型：{type(data)}\n")

            # 2. 根据数据类型写入内容
            if isinstance(data, list):
                f.write(f"📐 列表总长度：{len(data)}\n")
                f.write("=" * 80 + "\n")
                f.write("📝 详细数据（逐行显示）：\n")
                for i, item in enumerate(data):
                    f.write(f"  第 {i + 1} 条：{item}\n")
            elif isinstance(data, dict):
                f.write(f"🔑 字典的键：{list(data.keys())}\n")
                f.write("=" * 80 + "\n")
                f.write("📝 详细数据（按键显示）：\n")
                for key, value in data.items():
                    f.write(f"  键[{key}]：\n")
                    f.write(f"    类型：{type(value)}\n")
                    # 只预览前10条（避免数据量过大）
                    preview = value[:10] if isinstance(value, (list, tuple)) else value
                    f.write(f"    内容预览：{preview}\n")
            else:
                f.write(f"📝 数据内容：{data}\n")

        print(f"✅ {dataset_name} 已成功提取并保存至：{os.path.abspath(output_txt_path)}")

    except pickle.UnpicklingError:
        print(f"❌ 错误：{dataset_name} 不是有效的pickle格式文件")
    except Exception as e:
        print(f"❌ 处理 {dataset_name} 时出错：{str(e)}")


# ---------------------- 主程序：处理三个数据集 ----------------------
if __name__ == "__main__":
    # 1. 定义三个数据集的路径（假设脚本在HCRide-main根目录）
    base_data_dir = "data"
    datasets = [
        {
            "pkl": os.path.join(base_data_dir, "driver_preference.pkl"),
            "txt": "driver_preference_all.txt",
            "name": "司机偏好数据集 (driver_preference.pkl)"
        },
        {
            "pkl": os.path.join(base_data_dir, "driver_location.pkl"),
            "txt": "driver_location_all.txt",
            "name": "司机位置数据集 (driver_location.pkl)"
        },
        {
            "pkl": os.path.join(base_data_dir, "order.pkl"),
            "txt": "order_all.txt",
            "name": "订单数据集 (order.pkl)"
        }
    ]

    # 2. 逐个处理数据集
    print("开始提取三个数据集...\n")
    for ds in datasets:
        extract_pkl_to_txt(ds["pkl"], ds["txt"], ds["name"])
        print("-" * 50)

    print("\n🎉 所有数据集提取完成！请用记事本或代码编辑器打开生成的TXT文件查看。")