# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from collections import defaultdict
# from tqdm import tqdm
# import os
# import random

# # 设置随机种子以保证可复现性
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)


# def generate_scaffold(smiles, include_chirality=False):
#     """
#     为单个SMILES生成Murcko骨架。
#     如果无法解析，则返回SMILES本身作为回退（Fallback）。
#     """
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return smiles
#         scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
#         return scaffold
#     except Exception:
#         return smiles


# def scaffold_split_ddi(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
#     """
#     对DDI数据进行骨架划分。

#     Args:
#         data_list: list of [smiles1, smiles2, label]
#     Returns:
#         train_data, valid_data, test_data
#     """
#     print("正在提取所有药物的唯一SMILES...")

#     # 1. 提取所有唯一的药物SMILES
#     unique_drugs = set()
#     for s1, s2, _ in data_list:
#         unique_drugs.add(s1)
#         unique_drugs.add(s2)

#     print(f"共发现 {len(unique_drugs)} 种唯一药物。")

#     # 2. 计算每个药物的骨架
#     print("正在计算Murcko骨架...")

#     drug_to_scaffold = {}
#     scaffold_to_drugs = defaultdict(list)

#     for drug in tqdm(unique_drugs):
#         scaffold = generate_scaffold(drug)
#         drug_to_scaffold[drug] = scaffold
#         scaffold_to_drugs[scaffold].append(drug)

#     # 3. 按骨架大小排序（为了平衡划分）
#     scaffolds = list(scaffold_to_drugs.keys())
#     scaffolds.sort(key=lambda x: len(scaffold_to_drugs[x]), reverse=True)

#     # 4. 划分药物集合 (Drug Split)
#     train_drugs = set()
#     valid_drugs = set()
#     test_drugs = set()

#     n_total_drugs = len(unique_drugs)
#     train_cutoff = n_total_drugs * frac_train
#     valid_cutoff = n_total_drugs * (frac_train + frac_valid)

#     current_count = 0

#     print("正在划分药物集合...")

#     for scaff in scaffolds:
#         drugs_in_scaffold = scaffold_to_drugs[scaff]
#         n_scaff_drugs = len(drugs_in_scaffold)

#         if current_count < train_cutoff:
#             train_drugs.update(drugs_in_scaffold)
#         elif current_count < valid_cutoff:
#             valid_drugs.update(drugs_in_scaffold)
#         else:
#             test_drugs.update(drugs_in_scaffold)

#         current_count += n_scaff_drugs

#     print(f"药物划分结果: Train={len(train_drugs)}, Valid={len(valid_drugs)}, Test={len(test_drugs)}")

#     # 5. 划分DDI交互对 (Pair Split)
#     # 逻辑：
#     # Train: 仅包含 (Train_Drug, Train_Drug)
#     # Valid: 包含至少一个 Valid_Drug，且不包含 Test_Drug
#     # Test: 包含至少一个 Test_Drug

#     train_data = []
#     valid_data = []
#     test_data = []

#     print("正在根据药物归属划分交互对...")

#     for item in tqdm(data_list):
#         s1, s2, label = item

#         # 标记两个药物所属的集合
#         s1_in_test = s1 in test_drugs
#         s2_in_test = s2 in test_drugs

#         s1_in_valid = s1 in valid_drugs
#         s2_in_valid = s2 in valid_drugs

#         # 优先级：只要有测试集药物，就划入测试集（模拟最难的冷启动）
#         if s1_in_test or s2_in_test:
#             test_data.append(item)
#         # 其次，如果有验证集药物，划入验证集
#         elif s1_in_valid or s2_in_valid:
#             valid_data.append(item)
#         # 最后，剩下的就是两端都在训练集的
#         else:
#             train_data.append(item)

#     return train_data, valid_data, test_data


# def process_and_save(input_file, output_dir):
#     """
#     主处理函数
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 读取数据 (假设是CSV，无表头或有表头，这里按无表头处理，如果有表头请修改 header=0)
#     # 格式要求: smiles1, smiles2, label
#     print(f"正在读取文件: {input_file}")
#     try:
#         # 尝试读取CSV，假设有表头 header=0，列名为 smiles1, smiles2, label
#         # 如果你的csv没有表头，请改用 header=None，并使用 iloc 索引
#         df = pd.read_csv(input_file)

#         # 简单的列名检查，如果列名不对，尝试按位置读取
#         if 'smiles1' not in df.columns:
#             # 假设前三列是 s1, s2, label
#             print("未检测到标准列名，默认使用前三列...")
#             data_list = df.iloc[:, :3].values.tolist()
#         else:
#             data_list = df[['smiles1', 'smiles2', 'label']].values.tolist()

#     except Exception as e:
#         print(f"读取错误: {e}")
#         return

#     # 清洗数据：确保label是数字
#     cleaned_data = []
#     for row in data_list:
#         s1, s2, lbl = row
#         cleaned_data.append([str(s1), str(s2), int(float(lbl))])  # 确保格式正确

#     print(f"原始数据量: {len(cleaned_data)}")

#     # 执行骨架划分
#     train, valid, test = scaffold_split_ddi(cleaned_data)

#     print(f"划分完成:")
#     print(f"Train pairs: {len(train)}")
#     print(f"Valid pairs: {len(valid)}")
#     print(f"Test pairs:  {len(test)}")

#     # 保存为npy
#     print("正在保存为.npy 文件...")

#     np.save(os.path.join(output_dir, 'train.npy'), np.array(train))
#     np.save(os.path.join(output_dir, 'valid.npy'), np.array(valid))
#     np.save(os.path.join(output_dir, 'test.npy'), np.array(test))

#     print(f"所有文件已保存至 {output_dir}")


# # ================= 使用示例 =================
# if __name__ == "__main__":
#     # # 请将此处替换为你的数据集路径
#     # # 输入csv格式示例:
#     # # CCC..., NCCC..., 1
#     # # COC..., c1ccccc1..., 0
#     #
#     # INPUT_CSV = "drugbank.csv"  # 你的原始数据文件
#     # OUTPUT_DIR = "processed_data"  # 输出文件夹
#     #
#     # # 为了演示，如果文件不存在，我生成一个假的CSV
#     # if not os.path.exists(INPUT_CSV):
#     #     print("未找到输入文件，生成示例数据...")
#     #     data = {
#     #         'smiles1': ['CC(=O)OC1=CC=CC=C1C(=O)O'] * 50 + ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C'] * 50,  # 阿司匹林 & 咖啡因
#     #         'smiles2': ['CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'] * 50 + ['CCO'] * 50,  # 睾酮 & 乙醇
#     #         'label': [1] * 100  # 确保'label'列的长度与其他列一致
#     #     }
#     #     pd.DataFrame(data).to_csv(INPUT_CSV, index=False)
#     #
#     # process_and_save(INPUT_CSV, OUTPUT_DIR)
#     train_file = "processed_data/train.npy"
#     valid_file = "processed_data/valid.npy"
#     test_file = "processed_data/test.npy"

#     # 加载 .npy 文件
#     train_data = np.load(train_file, allow_pickle=True)  # allow_pickle=True 是为了允许加载包含 Python 对象的数据
#     valid_data = np.load(valid_file, allow_pickle=True)
#     test_data = np.load(test_file, allow_pickle=True)
#     train_data_10 = train_data[:256]
#     valid_data_10 = valid_data[:10]
#     test_data_10 = test_data[:10]
#     # 打印前10条数据以确认加载成功
#     print(f"Train data (first 10): {train_data_10}")

#     # 保存前10条数据为新的.npy文件
#     np.save('processed_data/train_test.npy', train_data_10)
#     np.save('processed_data/valid_test.npy', train_data_10)
#     np.save('processed_data/test_test.npy', train_data_10)
#     # 打印数据的一部分以确认加载成功
#     print(f"Train data: {train_data[:5]}")  # 打印前5条数据
#     print(f"Valid data: {valid_data[:5]}")
#     print(f"Test data: {test_data[:5]}")

# import pandas as pd
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# from collections import defaultdict, Counter
# from tqdm import tqdm
# import os
# import random

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)


# def generate_scaffold(smiles, include_chirality=False):
#     """为单个SMILES生成Murcko骨架"""
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return smiles
#         scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
#         return scaffold
#     except Exception:
#         return smiles


# def complement_missing_classes(source_data, target_data, target_labels, missing_labels, drug_to_scaffold, target_drugs):
#     """补全目标集缺失的类别（不破坏骨架约束）"""
#     complemented = []
#     for label in missing_labels:
#         candidates = []
#         for item in source_data:
#             s1, s2, lbl = item
#             if int(lbl) != label:
#                 continue
#             scaff1 = drug_to_scaffold[s1]
#             scaff2 = drug_to_scaffold[s2]
#             if (s1 in target_drugs and s2 in target_drugs) or (scaff1 not in [drug_to_scaffold[d] for d in train_drugs]):
#                 candidates.append(item)
#         if candidates:
#             target_data.append(candidates[0])
#             source_data.remove(candidates[0])
#             complemented.append(label)
#     return source_data, target_data, complemented


# def supplement_min_samples(data, label_to_pairs, all_assigned, min_samples=2):
#     """补充少数类别样本（确保≥min_samples）"""
#     label_counter = Counter([int(item[2]) for item in data])
#     supplemented = []
#     for label, count in label_counter.items():
#         if count >= min_samples:
#             continue
#         needed = min_samples - count
#         # Convert to tuples for comparison
#         candidates = [p for pairs in label_to_pairs.values() for p in pairs 
#                       if int(p[2]) == label and tuple(p) not in all_assigned]
#         if candidates:
#             add_num = min(needed, len(candidates))
#             data.extend(candidates[:add_num])
#             # Add as tuples to the set
#             for item in candidates[:add_num]:
#                 all_assigned.add(tuple(item))
#             supplemented.append((label, add_num))
#     return data, supplemented, all_assigned


# def adjust_split_ratio(train_data, valid_data, test_data, frac_train, frac_valid, frac_test, label_to_pairs):
#     """微调数据比例（贴近目标值）"""
#     total = len(train_data) + len(valid_data) + len(test_data)
#     target_train = int(total * frac_train)
#     target_valid = int(total * frac_valid)
#     target_test = int(total * frac_test)
    
#     # Create set with tuples
#     all_assigned = set()
#     for item in train_data + valid_data + test_data:
#         all_assigned.add(tuple(item))
    
#     # 调整训练集
#     if len(train_data) > target_train:
#         train_counter = Counter([int(item[2]) for item in train_data])
#         new_train = []
#         for label in train_counter.keys():
#             label_samples = [item for item in train_data if int(item[2]) == label]
#             keep_ratio = target_train / len(train_data)
#             keep_num = max(int(len(label_samples) * keep_ratio), 2)
#             new_train.extend(label_samples[:keep_num])
#         train_data = new_train
#     elif len(train_data) < target_train:
#         remaining = [p for pairs in label_to_pairs.values() for p in pairs 
#                     if tuple(p) not in all_assigned]
#         add_num = target_train - len(train_data)
#         train_data.extend(remaining[:add_num])
#         for item in remaining[:add_num]:
#             all_assigned.add(tuple(item))
    
#     # 调整验证集
#     if len(valid_data) > target_valid:
#         counter = Counter([int(item[2]) for item in valid_data])
#         new_data = []
#         for label in counter.keys():
#             label_samples = [item for item in valid_data if int(item[2]) == label]
#             keep_num = max(int(len(label_samples) * target_valid / len(valid_data)), 2)
#             new_data.extend(label_samples[:keep_num])
#         valid_data = new_data
#     elif len(valid_data) < target_valid:
#         remaining = [p for pairs in label_to_pairs.values() for p in pairs 
#                     if tuple(p) not in all_assigned]
#         add_num = target_valid - len(valid_data)
#         valid_data.extend(remaining[:add_num])
#         for item in remaining[:add_num]:
#             all_assigned.add(tuple(item))
    
#     # 调整测试集
#     if len(test_data) > target_test:
#         counter = Counter([int(item[2]) for item in test_data])
#         new_data = []
#         for label in counter.keys():
#             label_samples = [item for item in test_data if int(item[2]) == label]
#             keep_num = max(int(len(label_samples) * target_test / len(test_data)), 2)
#             new_data.extend(label_samples[:keep_num])
#         test_data = new_data
#     elif len(test_data) < target_test:
#         remaining = [p for pairs in label_to_pairs.values() for p in pairs 
#                     if tuple(p) not in all_assigned]
#         add_num = target_test - len(test_data)
#         test_data.extend(remaining[:add_num])
#         for item in remaining[:add_num]:
#             all_assigned.add(tuple(item))
    
#     return train_data, valid_data, test_data


# def scaffold_stratified_split_ddi(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, min_samples_per_class=2):
#     """
#     混合策略：骨架切分 + 分层抽样 + 类别补全 + 样本均衡
#     """
#     print("="*60)
#     print("开始混合策略划分（骨架切分 + 分层抽样 + 类别补全）")
#     print("="*60)
    
#     # ==================== Step 1: 分析标签分布 ====================
#     print("\n[Step 1] 分析标签分布...")
#     labels = [int(item[2]) for item in data_list]
#     label_counter = Counter(labels)
#     all_labels = set(range(86))  # DDI多分类固定86类
    
#     print(f"总样本数: {len(data_list)}")
#     print(f"类别数: {len(label_counter)}")
#     print(f"最多样本的类别: {label_counter.most_common(1)}")
#     print(f"最少样本的类别: {label_counter.most_common()[-1]}")
    
#     rare_classes = {k: v for k, v in label_counter.items() 
#                     if v < min_samples_per_class * 3}
#     if rare_classes:
#         print(f"⚠️ 发现 {len(rare_classes)} 个稀有类别（样本数 < {min_samples_per_class * 3}）:")
#         for cls, cnt in sorted(rare_classes.items(), key=lambda x: x[1])[:5]:
#             print(f"  - 类别 {cls}: {cnt} 样本")
    
#     # ==================== Step 2: 骨架划分药物 ====================
#     print("\n[Step 2] 计算药物骨架...")
#     unique_drugs = set()
#     for s1, s2, _ in data_list:
#         unique_drugs.add(s1)
#         unique_drugs.add(s2)
    
#     drug_to_scaffold = {}
#     scaffold_to_drugs = defaultdict(list)
    
#     for drug in tqdm(unique_drugs, desc="生成骨架"):
#         scaffold = generate_scaffold(drug)
#         drug_to_scaffold[drug] = scaffold
#         scaffold_to_drugs[scaffold].append(drug)
    
#     print(f"唯一药物数: {len(unique_drugs)}")
#     print(f"唯一骨架数: {len(scaffold_to_drugs)}")
    
#     # 打乱骨架顺序，避免大骨架集中
#     scaffolds = list(scaffold_to_drugs.keys())
#     random.shuffle(scaffolds)
    
#     # 初步划分药物
#     global train_drugs, valid_drugs, test_drugs
#     train_drugs = set()
#     valid_drugs = set()
#     test_drugs = set()
    
#     n_total = len(unique_drugs)
#     train_cutoff = n_total * frac_train
#     valid_cutoff = n_total * (frac_train + frac_valid)
    
#     current = 0
#     for scaff in scaffolds:
#         drugs = scaffold_to_drugs[scaff]
#         if current < train_cutoff:
#             train_drugs.update(drugs)
#         elif current < valid_cutoff:
#             valid_drugs.update(drugs)
#         else:
#             test_drugs.update(drugs)
#         current += len(drugs)
    
#     print(f"初步药物划分: Train={len(train_drugs)}, Valid={len(valid_drugs)}, Test={len(test_drugs)}")
    
#     # ==================== Step 3: 按类别组织数据 ====================
#     print("\n[Step 3] 按类别组织交互对...")
#     label_to_pairs = defaultdict(list)
#     for item in data_list:
#         label = int(item[2])
#         label_to_pairs[label].append(item)
    
#     # ==================== Step 4: 初步分配 ====================
#     print("\n[Step 4] 执行初步分配...")
#     train_data = []
#     valid_data = []
#     test_data = []
#     flexible = []
    
#     for label, pairs in tqdm(label_to_pairs.items(), desc="处理类别"):
#         for pair in pairs:
#             s1, s2, lbl = pair
#             in_train = (s1 in train_drugs and s2 in train_drugs)
#             in_valid = ((s1 in valid_drugs or s2 in valid_drugs) and 
#                        s1 not in test_drugs and s2 not in test_drugs)
#             in_test = (s1 in test_drugs or s2 in test_drugs)
            
#             if in_test:
#                 test_data.append(pair)
#             elif in_valid:
#                 valid_data.append(pair)
#             elif in_train:
#                 train_data.append(pair)
#             else:
#                 flexible.append(pair)
    
#     # ==================== Step 5: 补全缺失类别 ====================
#     print("\n[Step 5] 补全缺失类别...")
#     train_labels = set([int(item[2]) for item in train_data])
#     valid_labels = set([int(item[2]) for item in valid_data])
#     test_labels = set([int(item[2]) for item in test_data])
    
#     # 补全Valid集
#     missing_valid = all_labels - valid_labels
#     if missing_valid:
#         train_data, valid_data, comp_valid = complement_missing_classes(
#             train_data, valid_data, valid_labels, missing_valid, drug_to_scaffold, valid_drugs
#         )
#         print(f"Valid集补全类别: {comp_valid}，剩余缺失: {len(all_labels - set([int(item[2]) for item in valid_data]))}")
    
#     # 补全Test集
#     missing_test = all_labels - test_labels
#     if missing_test:
#         train_data, test_data, comp_test1 = complement_missing_classes(
#             train_data, test_data, test_labels, missing_test, drug_to_scaffold, test_drugs
#         )
#         valid_data, test_data, comp_test2 = complement_missing_classes(
#             valid_data, test_data, test_labels, missing_test - set(comp_test1), drug_to_scaffold, test_drugs
#         )
#         print(f"Test集补全类别: {comp_test1 + comp_test2}，剩余缺失: {len(all_labels - set([int(item[2]) for item in test_data]))}")
    
#     # 补全Train集
#     missing_train = all_labels - train_labels
#     if missing_train:
#         for label in missing_train:
#             candidates = [p for pairs in label_to_pairs.values() for p in pairs 
#                           if int(p[2]) == label and p not in train_data+valid_data+test_data+flexible]
#             if candidates:
#                 train_data.append(candidates[0])
    
#     # ==================== Step 6: 补充少数类别样本 ====================
#     print("\n[Step 6] 补充少数类别样本...")
#     # Create set with tuples
#     all_assigned = set()
#     for item in train_data + valid_data + test_data + flexible:
#         all_assigned.add(tuple(item))
    
#     valid_data, supp_valid, all_assigned = supplement_min_samples(
#         valid_data, label_to_pairs, all_assigned, min_samples_per_class
#     )
#     test_data, supp_test, all_assigned = supplement_min_samples(
#         test_data, label_to_pairs, all_assigned, min_samples_per_class
#     )
#     train_data, supp_train, all_assigned = supplement_min_samples(
#         train_data, label_to_pairs, all_assigned, min_samples_per_class
#     )
#     print(f"Train集补充: {supp_train}")
#     print(f"Valid集补充: {supp_valid}")
#     print(f"Test集补充: {supp_test}")
    
#     # ==================== Step 7: 微调数据比例 ====================
#     print("\n[Step 7] 微调数据比例...")
#     train_data, valid_data, test_data = adjust_split_ratio(
#         train_data, valid_data, test_data, frac_train, frac_valid, frac_test, label_to_pairs
#     )
    
#     # ==================== Step 8: 验证最终分布 ====================
#     print("\n[Step 8] 验证最终分布...")
    
#     def analyze_split(data, name):
#         labels = [int(item[2]) for item in data]
#         counter = Counter(labels)
#         print(f"\n{name}:")
#         print(f"  总样本: {len(data)}")
#         print(f"  唯一类别: {len(counter)}")
#         print(f"  类别范围: [{min(labels) if labels else 'N/A'}, {max(labels) if labels else 'N/A'}]")
        
#         missing = all_labels - set(labels)
#         if missing:
#             print(f"  ⚠️ 缺失类别数: {len(missing)}")
#             print(f"     缺失类别: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
#         else:
#             print(f"  ✅ 所有86个类别都存在")
        
#         rare = {k: v for k, v in counter.items() if v < min_samples_per_class}
#         if rare:
#             print(f"  ⚠️ 样本数 < {min_samples_per_class} 的类别: {len(rare)}")
        
#         return counter
    
#     train_counter = analyze_split(train_data, "Train")
#     valid_counter = analyze_split(valid_data, "Valid")
#     test_counter = analyze_split(test_data, "Test")
    
#     # 比例检查
#     total = len(train_data) + len(valid_data) + len(test_data)
#     print(f"\n实际比例: Train={len(train_data)/total:.2%}, Valid={len(valid_data)/total:.2%}, Test={len(test_data)/total:.2%}")
#     print(f"目标比例: Train={frac_train:.2%}, Valid={frac_valid:.2%}, Test={frac_test:.2%}")
    
#     return train_data, valid_data, test_data


# def validate_and_clean_data(data_list):
#     """验证并清洗数据"""
#     cleaned = []
#     invalid_smiles = 0
#     invalid_labels = 0
    
#     for s1, s2, label in tqdm(data_list, desc="验证数据"):
#         # 检查 SMILES
#         mol1 = Chem.MolFromSmiles(str(s1))
#         mol2 = Chem.MolFromSmiles(str(s2))
        
#         if mol1 is None or mol2 is None:
#             invalid_smiles += 1
#             continue
        
#         # 检查标签（DDI多分类固定0-85）
#         try:
#             label = int(float(label))
#             if not (0 <= label < 86):
#                 invalid_labels += 1
#                 continue
#         except:
#             invalid_labels += 1
#             continue
        
#         cleaned.append([str(s1), str(s2), label])
    
#     print(f"移除无效SMILES: {invalid_smiles}")
#     print(f"移除无效标签: {invalid_labels}")
#     print(f"保留样本: {len(cleaned)}")
    
#     return cleaned


# def process_and_save(input_file, output_dir, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
#     """主处理函数"""
#     os.makedirs(output_dir, exist_ok=True)
    
#     print(f"读取文件: {input_file}")
#     df = pd.read_csv(input_file)
    
#     # 读取数据
#     if 'smiles1' in df.columns and 'smiles2' in df.columns and 'label' in df.columns:
#         data_list = df[['smiles1', 'smiles2', 'label']].values.tolist()
#     else:
#         print("未检测到标准列名，使用前3列...")
#         data_list = df.iloc[:, :3].values.tolist()
    
#     print(f"原始数据量: {len(data_list)}")
    
#     # 清洗数据
#     cleaned_data = validate_and_clean_data(data_list)
    
#     # 执行混合策略划分
#     train, valid, test = scaffold_stratified_split_ddi(
#         cleaned_data,
#         frac_train=frac_train,
#         frac_valid=frac_valid,
#         frac_test=frac_test,
#         min_samples_per_class=2
#     )
    
#     # 保存
#     print(f"\n保存文件到 {output_dir}...")
#     np.save(os.path.join(output_dir, 'train_sp.npy'), np.array(train, dtype=object))
#     np.save(os.path.join(output_dir, 'valid_sp.npy'), np.array(valid, dtype=object))
#     np.save(os.path.join(output_dir, 'test_sp.npy'), np.array(test, dtype=object))
    
#     print("✅ 处理完成！")


# def stratified_sample(data, n_samples=256):
#     """分层采样，确保每个类别都有样本"""
#     label_to_indices = defaultdict(list)
    
#     for i in range(len(data)):
#         label = int(data[i][2])
#         label_to_indices[label].append(i)
    
#     sampled_indices = []
#     n_per_class = max(1, n_samples // len(label_to_indices))
    
#     for label, indices in label_to_indices.items():
#         sampled_indices.extend(indices[:n_per_class])
    
#     if len(sampled_indices) < n_samples:
#         all_indices = set(range(len(data)))
#         remaining_indices = list(all_indices - set(sampled_indices))
#         random.shuffle(remaining_indices)
#         sampled_indices.extend(remaining_indices[:n_samples - len(sampled_indices)])
    
#     sampled_indices = sampled_indices[:n_samples]
#     sampled = [data[i].tolist() if isinstance(data[i], np.ndarray) else data[i] 
#                for i in sampled_indices]
    
#     return np.array(sampled, dtype=object)


# if __name__ == "__main__":
#     INPUT_CSV = "drugbank.csv"
#     OUTPUT_DIR = "processed_data"
    
#     # 处理完整数据集
#     process_and_save(INPUT_CSV, OUTPUT_DIR)
    
#     # 创建小样本用于快速测试
#     print("\n" + "="*60)
#     print("创建测试子集...")
#     print("="*60)
    
#     train = np.load('processed_data/train_sp.npy', allow_pickle=True)
#     valid = np.load('processed_data/valid_sp.npy', allow_pickle=True)
#     test = np.load('processed_data/test_sp.npy', allow_pickle=True)
    
#     train_test = stratified_sample(train, 512)
#     valid_test = stratified_sample(valid, 256)
#     test_test = stratified_sample(test, 256)
    
#     np.save('processed_data/train_test_sp.npy', train_test)
#     np.save('processed_data/valid_test_sp.npy', valid_test)
#     np.save('processed_data/test_test_sp.npy', test_test)
    
#     print(f"Train test: {len(train_test)} samples")
#     print(f"Valid test: {len(valid_test)} samples")
#     print(f"Test test: {len(test_test)} samples")
    
#     # 验证分布
#     print("\n验证测试子集的类别分布:")
#     for name, data in [("Train", train_test), ("Valid", valid_test), ("Test", test_test)]:
#         labels = [int(item[2]) for item in data]
#         unique_labels = len(set(labels))
#         print(f"{name}: {len(data)} 样本, {unique_labels} 个唯一类别")
    
#     print("\n✅ 全部完成！")

import pandas as pd
import numpy as np
from rdkit import Chem
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def stratified_split_ddi(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1, min_samples_per_class=2):
    """
    仅使用标签分布的分层采样划分 DDI 数据（不采用骨架切分）

    思路：
    1. 按类别(label)把样本分组。
    2. 每个类别内部打乱，然后按比例切分到 train/valid/test。
    3. 对于样本数过少的类别，会尽量保证总划分不超过该类别总样本。
    """
    print("=" * 60)
    print("开始分层抽样划分（不使用骨架切分）")
    print("=" * 60)

    # ==================== Step 1: 分析标签分布 ====================
    print("\n[Step 1] 分析标签分布...")
    labels = [int(item[2]) for item in data_list]
    label_counter = Counter(labels)

    print(f"总样本数: {len(data_list)}")
    print(f"类别数: {len(label_counter)}")
    print(f"最多样本的类别: {label_counter.most_common(1)}")
    print(f"最少样本的类别: {label_counter.most_common()[-1]}")

    rare_classes = {k: v for k, v in label_counter.items()
                    if v < min_samples_per_class * 3}  # 三个集合的下限
    if rare_classes:
        print(f"⚠️ 发现 {len(rare_classes)} 个稀有类别（样本数 < {min_samples_per_class * 3}）:")
        for cls, cnt in sorted(rare_classes.items(), key=lambda x: x[1])[:10]:
            print(f"  - 类别 {cls}: {cnt} 样本")

    # ==================== Step 2: 按类别分组 ====================
    print("\n[Step 2] 按类别分组样本...")
    label_to_pairs = defaultdict(list)
    for item in data_list:
        label = int(item[2])
        label_to_pairs[label].append(item)

    # ==================== Step 3: 分层划分 ====================
    print("\n[Step 3] 执行分层划分...")
    train_data, valid_data, test_data = [], [], []

    for label, pairs in tqdm(label_to_pairs.items(), desc="按类别分配"):
        total = len(pairs)
        random.shuffle(pairs)

        # 初始目标数量
        target_train = int(total * frac_train)
        target_valid = int(total * frac_valid)
        target_test = total - target_train - target_valid

        # 保证每个集合至少有一点，但不能超过总数
        # 可以根据需要决定要不要强行 min_samples_per_class
        # 这里采用“尽量保证”的策略，但不强制超过 total
        # 防止 target_sum > total 的情况
        if total >= min_samples_per_class * 3:
            target_train = max(target_train, min_samples_per_class)
            target_valid = max(target_valid, min_samples_per_class)
            target_test = max(target_test, min_samples_per_class)

            # 如果超过总数，则按比例回缩
            if target_train + target_valid + target_test > total:
                scale = total / (target_train + target_valid + target_test)
                target_train = int(target_train * scale)
                target_valid = int(target_valid * scale)
                target_test = total - target_train - target_valid
        else:
            # 样本太少时，直接按比例切分，不强求每个集合都有
            target_train = int(total * frac_train)
            target_valid = int(total * frac_valid)
            target_test = total - target_train - target_valid

        # 划分
        train_data.extend(pairs[:target_train])
        valid_data.extend(pairs[target_train:target_train + target_valid])
        test_data.extend(pairs[target_train + target_valid:])

    # ==================== Step 4: 验证分布 ====================
    print("\n[Step 4] 验证最终分布...")

    def analyze_split(data, name):
        labels = [int(item[2]) for item in data]
        counter = Counter(labels)
        print(f"\n{name}:")
        print(f"  总样本: {len(data)}")
        print(f"  唯一类别: {len(counter)}")

        if len(counter) > 0:
            print(f"  样本最多的类别: {counter.most_common(1)}")
            print(f"  样本最少的类别: {counter.most_common()[-1]}")

        # 如果你确定是 0~85 共 86 类，可以打开下面这一段检查缺失类别
        try:
            expected = set(range(86))
            actual = set(counter.keys())
            missing = expected - actual
            if missing:
                print(f"  ⚠️ 缺失类别数: {len(missing)}")
                print(f"     缺失类别: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            else:
                print(f"  ✅ 所有86个类别都存在")
        except Exception:
            pass

        # 样本非常少的类别提示
        rare = {k: v for k, v in counter.items() if v < min_samples_per_class}
        if rare:
            print(f"  ⚠️ 样本数 < {min_samples_per_class} 的类别: {len(rare)}")

        return counter

    train_counter = analyze_split(train_data, "Train")
    valid_counter = analyze_split(valid_data, "Valid")
    test_counter = analyze_split(test_data, "Test")

    total_all = len(train_data) + len(valid_data) + len(test_data)
    print("\n[Step 5] 全局比例检查:")
    print(f"实际比例: Train={len(train_data)/total_all:.2%}, Valid={len(valid_data)/total_all:.2%}, Test={len(test_data)/total_all:.2%}")
    print(f"目标比例: Train={frac_train:.2%}, Valid={frac_valid:.2%}, Test={frac_test:.2%}")

    return train_data, valid_data, test_data


def validate_and_clean_data(data_list):
    """验证并清洗数据"""
    cleaned = []
    invalid_smiles = 0
    invalid_labels = 0

    for s1, s2, label in tqdm(data_list, desc="验证数据"):
        # 检查 SMILES
        mol1 = Chem.MolFromSmiles(str(s1))
        mol2 = Chem.MolFromSmiles(str(s2))

        if mol1 is None or mol2 is None:
            invalid_smiles += 1
            continue

        # 检查标签
        try:
            label = int(float(label))
            # 这里假设标签范围为 [0, 86)，你可以根据自己数据修改
            if not (0 <= label < 86):
                invalid_labels += 1
                continue
        except Exception:
            invalid_labels += 1
            continue

        cleaned.append([str(s1), str(s2), label])

    print(f"移除无效SMILES: {invalid_smiles}")
    print(f"移除无效标签: {invalid_labels}")
    print(f"保留样本: {len(cleaned)}")

    return cleaned


def process_and_save(input_file, output_dir, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    """主处理函数"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file)

    # 读取数据
    if 'smiles1' in df.columns:
        data_list = df[['smiles1', 'smiles2', 'label']].values.tolist()
    else:
        print("未检测到标准列名，使用前3列...")
        data_list = df.iloc[:, :3].values.tolist()

    print(f"原始数据量: {len(data_list)}")

    # 清洗数据
    cleaned_data = validate_and_clean_data(data_list)

    # 执行“纯分层抽样”划分
    train, valid, test = stratified_split_ddi(
        cleaned_data,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        min_samples_per_class=2  # 可根据需要调整
    )

    # 保存
    print(f"\n保存文件到 {output_dir}...")
    np.save(os.path.join(output_dir, 'train.npy'), np.array(train, dtype=object))
    np.save(os.path.join(output_dir, 'valid.npy'), np.array(valid, dtype=object))
    np.save(os.path.join(output_dir, 'test.npy'), np.array(test, dtype=object))

    print("✅ 处理完成！")


def stratified_sample(data, n_samples=256):
    """分层采样，确保每个类别都有样本"""
    label_to_indices = defaultdict(list)

    # 收集每个类别的索引
    for i in range(len(data)):
        label = int(data[i][2])
        label_to_indices[label].append(i)

    sampled_indices = []
    n_per_class = max(1, n_samples // len(label_to_indices))

    # 从每个类别采样
    for label, indices in label_to_indices.items():
        # 为避免类别内部偏向，先打乱索引
        idx_list = indices.copy()
        random.shuffle(idx_list)
        sampled_indices.extend(idx_list[:n_per_class])

    # 如果还不够，随机补充
    if len(sampled_indices) < n_samples:
        all_indices = set(range(len(data)))
        remaining_indices = list(all_indices - set(sampled_indices))
        random.shuffle(remaining_indices)
        sampled_indices.extend(remaining_indices[:n_samples - len(sampled_indices)])

    # 根据索引提取样本
    sampled_indices = sampled_indices[:n_samples]
    sampled = [data[i].tolist() if isinstance(data[i], np.ndarray) else data[i]
               for i in sampled_indices]

    return np.array(sampled, dtype=object)


if __name__ == "__main__":
    INPUT_CSV = "drugbank.csv"
    OUTPUT_DIR = "processed_data"

    # 处理完整数据集
    process_and_save(INPUT_CSV, OUTPUT_DIR)

    # 创建小样本用于快速测试
    print("\n" + "=" * 60)
    print("创建测试子集...")
    print("=" * 60)

    train = np.load('processed_data/train.npy', allow_pickle=True)
    valid = np.load('processed_data/valid.npy', allow_pickle=True)
    test = np.load('processed_data/test.npy', allow_pickle=True)

    train_test = stratified_sample(train, 512)
    valid_test = stratified_sample(valid, 256)
    test_test = stratified_sample(test, 256)

    np.save('processed_data/train_test.npy', train_test)
    np.save('processed_data/valid_test.npy', valid_test)
    np.save('processed_data/test_test.npy', test_test)

    print(f"Train test: {len(train_test)} samples")
    print(f"Valid test: {len(valid_test)} samples")
    print(f"Test test: {len(test_test)} samples")

    # 验证分布
    print("\n验证测试子集的类别分布:")
    for name, data in [("Train", train_test), ("Valid", valid_test), ("Test", test_test)]:
        labels = [int(item[2]) for item in data]
        unique_labels = len(set(labels))
        print(f"{name}: {len(data)} 样本, {unique_labels} 个唯一类别")

    print("\n✅ 全部完成！")
