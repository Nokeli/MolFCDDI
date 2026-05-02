import pandas as pd
import numpy as np

def csv_to_npy_with_pandas(csv_file_path, npy_file_path):
    """
    读取包含smiles1、smiles2、label列的CSV文件，并保存为npy文件
    :param csv_file_path: 输入的CSV文件路径
    :param npy_file_path: 输出的npy文件路径
    """
    try:
        # 1. 读取CSV文件（自动识别表头）
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        
        # 验证列是否存在（避免列名错误）
        required_columns = ['smiles1', 'smiles2', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要列：{missing_columns}")
        
        # 2. 转换为numpy数组（保留结构化信息，也可拆分为单独数组）
        # 方式1：保存为结构化数组（推荐，可保留列名对应关系）
        data_array = df[required_columns].to_records(index=False)
        # 方式2：保存为普通二维数组（纯数据，无列名）
        # data_array = df[required_columns].values
        
        # 3. 保存为npy文件
        np.save(npy_file_path, data_array)
        
        print(f"成功！CSV文件已转换为npy文件：{npy_file_path}")
        print(f"数据形状：{data_array.shape}")
        print(f"前3行数据预览：")
        print(data_array[:3])
        
        return data_array
        
    except FileNotFoundError:
        print(f"错误：找不到CSV文件 {csv_file_path}")
    except ValueError as e:
        print(f"数据格式错误：{e}")
    except Exception as e:
        print(f"转换失败：{e}")

# 调用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    csv_path = "ddi_test.csv"  # 输入CSV文件
    npy_path = "ddi_test.npy"  # 输出npy文件
    csv_to_npy_with_pandas(csv_path, npy_path)