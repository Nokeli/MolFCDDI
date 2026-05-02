#!/usr/bin/env python3
"""
将药物相互作用数据从 (drug1_id, type, drug2_id) 格式
转换为 (smiles1, smiles2, label) 格式
"""

def load_smiles_data(smiles_file):
    """加载SMILES数据,返回 drug_id -> smiles 的字典"""
    smiles_dict = {}
    
    with open(smiles_file, 'r', encoding='utf-8') as f:
        # 跳过表头
        header = f.readline().strip()
        print(f"  表头: {header}")
        
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            
            # 尝试不同的分隔符
            parts = None
            if '\t' in line:
                parts = line.split('\t', 1)  # 用制表符分隔,只分两部分
            elif ',' in line:
                # 对于逗号,先找到第一个逗号的位置
                comma_pos = line.index(',')
                drug_id = line[:comma_pos].strip()
                smiles = line[comma_pos+1:].strip()
                parts = [drug_id, smiles]
            else:
                # 如果是空格分隔,取第一个字段作为drug_id,剩余作为smiles
                parts = line.split(None, 1)
            
            if parts and len(parts) == 2:
                drug_id = parts[0].strip()
                smiles = parts[1].strip()
                
                # 验证drug_id格式
                if drug_id.startswith('DB') and len(drug_id) == 7:
                    smiles_dict[drug_id] = smiles
                    if line_num <= 5:  # 只显示前几个
                        print(f"  加载: {drug_id} -> SMILES长度 {len(smiles)}")
                else:
                    print(f"  警告: 第{line_num}行 - 无效的drug_id格式: {drug_id}")
            else:
                print(f"  警告: 第{line_num}行 - 无法解析: {line[:50]}...")
    
    return smiles_dict

def convert_interactions(interaction_file, smiles_dict, output_file):
    """转换相互作用数据格式"""
    
    converted_count = 0
    skipped_count = 0
    missing_drugs = set()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 写入表头
        out_f.write("smiles1,smiles2,label\n")
        
        with open(interaction_file, 'r', encoding='utf-8') as in_f:
            for line_num, line in enumerate(in_f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                # 解析行: drug1_id,type,drug2_id
                parts = line.split(',')
                if len(parts) != 3:
                    print(f"警告: 第{line_num}行格式错误,跳过: {line}")
                    skipped_count += 1
                    continue
                
                drug1_id = parts[0].strip()
                label = parts[1].strip()
                drug2_id = parts[2].strip()
                
                # 查找对应的SMILES
                if drug1_id in smiles_dict and drug2_id in smiles_dict:
                    smiles1 = smiles_dict[drug1_id]
                    smiles2 = smiles_dict[drug2_id]
                    
                    # 写入新格式 (用双引号包围SMILES,防止内部逗号干扰)
                    out_f.write(f'"{smiles1}","{smiles2}",{label}\n')
                    converted_count += 1
                else:
                    # 记录缺失的药物ID
                    if drug1_id not in smiles_dict:
                        missing_drugs.add(drug1_id)
                    if drug2_id not in smiles_dict:
                        missing_drugs.add(drug2_id)
                    skipped_count += 1
    
    # 显示缺失的药物ID
    if missing_drugs:
        print(f"\n缺失SMILES数据的药物ID ({len(missing_drugs)}个):")
        for drug_id in sorted(missing_drugs)[:10]:  # 只显示前10个
            print(f"  - {drug_id}")
        if len(missing_drugs) > 10:
            print(f"  ... 还有 {len(missing_drugs)-10} 个")
    
    return converted_count, skipped_count

def main():
    import sys
    
    # 文件路径 - 可以通过命令行参数指定
    if len(sys.argv) == 4:
        interaction_file = sys.argv[1]
        smiles_file = sys.argv[2]
        output_file = sys.argv[3]
    else:
        interaction_file = "ddi_test1xiao.csv"  # 第一个文件:药物相互作用数据
        smiles_file = "drug_smiles.csv"  # 第二个文件:药物SMILES数据
        output_file = "ddi_test.csv"  # 输出文件
    
    print("="*60)
    print("药物相互作用数据格式转换工具")
    print("="*60)
    print(f"\n输入文件:")
    print(f"  - 相互作用数据: {interaction_file}")
    print(f"  - SMILES数据: {smiles_file}")
    print(f"输出文件:")
    print(f"  - {output_file}")
    print()
    
    print("正在加载SMILES数据...")
    smiles_dict = load_smiles_data(smiles_file)
    print(f"✓ 已加载 {len(smiles_dict)} 个药物的SMILES数据")
    
    print("\n正在转换相互作用数据...")
    converted, skipped = convert_interactions(interaction_file, smiles_dict, output_file)
    
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"✓ 成功转换: {converted} 条记录")
    print(f"✗ 跳过: {skipped} 条记录")
    print(f"✓ 输出文件: {output_file}")
    print()

if __name__ == "__main__":
    main()