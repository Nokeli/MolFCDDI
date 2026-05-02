"""
Cold Start DDI Dataset Loader
解析 CSV Neg samples 列，构造 pos/neg pair
"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from chemprop.features import BatchMolGraph, MolGraph


class ColdStartDataset(Dataset):
    """
    冷启动数据集：每个 sample = {'pos': {...}, 'neg': {...}}
    从 CSV 解析 Neg samples 列（drug_id$flag）构造负样本
    """
    def __init__(self, csv_path: str, drug_smiles_path: str):
        # 加载 drug_id → smiles 映射
        df_smiles = pd.read_csv(drug_smiles_path)
        self.drug_smiles = dict(zip(df_smiles['drug_id'], df_smiles['smiles']))
        
        # 加载 DDI CSV
        self.df = pd.read_csv(csv_path)
        
        # 解析 pos / neg pairs
        self.samples = []
        for _, row in self.df.iterrows():
            d1, d2, rel_type = row['d1'], row['d2'], int(row['type'])
            # 注意：CSV 中 neg sample 实际在 'split' 列，'Neg samples' 列为空
            neg_str = str(row['split']) if pd.notna(row.get('split')) else str(row.get('Neg samples', ''))
            
            # 解析 Neg samples
            if '$' not in neg_str:
                continue
            
            neg_drug, flag = neg_str.split('$')
            
            # 构造 pos pair
            pos = {
                'smiles1': self.drug_smiles[d1],
                'smiles2': self.drug_smiles[d2],
                'drug1': d1, 'drug2': d2,
                'rel_type': rel_type
            }
            
            # 构造 neg pair（根据 flag 替换 head 或 tail）
            if flag == 't':
                neg = {
                    'smiles1': self.drug_smiles[d1],
                    'smiles2': self.drug_smiles.get(neg_drug, ''),
                    'drug1': d1, 'drug2': neg_drug,
                    'rel_type': rel_type
                }
            else:  # flag == 'h'
                neg = {
                    'smiles1': self.drug_smiles.get(neg_drug, ''),
                    'smiles2': self.drug_smiles[d2],
                    'drug1': neg_drug, 'drug2': d2,
                    'rel_type': rel_type
                }
            
            # 跳过无效 smiles
            if not neg['smiles1'] or not neg['smiles2']:
                continue
            
            self.samples.append({'pos': pos, 'neg': neg})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class ColdStartCollator:
    """
    将 list of {'pos': {...}, 'neg': {...}} 转换为 model 输入
    复用 BatchMolGraph 批处理逻辑
    """
    def __init__(self, args, cache_size: int = 10000):
        self.args = args
        self.graph_cache = {}
        self.cache_order = []
        self.cache_size = cache_size
    
    def _get_graph(self, smiles: str):
        """LRU cache: smiles → MolGraph"""
        if smiles in self.graph_cache:
            # 更新 LRU 顺序
            self.cache_order.remove(smiles)
            self.cache_order.append(smiles)
            return self.graph_cache[smiles]
        
        # 创建新图
        mol_graph = MolGraph(smiles, self.args, pretrain=False, brics2emb=None)
        
        # 添加到缓存
        self.graph_cache[smiles] = mol_graph
        self.cache_order.append(smiles)
        
        # LRU 清理
        if len(self.cache_order) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.graph_cache[oldest]
        
        return mol_graph
    
    def _batch_graphs(self, smiles_list):
        """批量构造图"""
        mol_graphs = [self._get_graph(s) for s in smiles_list]
        return BatchMolGraph(mol_graphs, [], self.args)
    
    def __call__(self, batch):
        """
        Args:
            batch: list of {'pos': {...}, 'neg': {...}}
        Returns:
            dict with 'pos', 'neg', 'rel_types'
        """
        # 收集 pos / neg smiles
        pos_smiles1 = [item['pos']['smiles1'] for item in batch]
        pos_smiles2 = [item['pos']['smiles2'] for item in batch]
        neg_smiles1 = [item['neg']['smiles1'] for item in batch]
        neg_smiles2 = [item['neg']['smiles2'] for item in batch]
        
        # 构造 batch graphs
        pos_graph1 = self._batch_graphs(pos_smiles1)
        pos_graph2 = self._batch_graphs(pos_smiles2)
        neg_graph1 = self._batch_graphs(neg_smiles1)
        neg_graph2 = self._batch_graphs(neg_smiles2)
        
        rel_types = [item['pos']['rel_type'] for item in batch]
        
        return {
            'pos': (pos_graph1, pos_graph2),
            'neg': (neg_graph1, neg_graph2),
            'rel_types': rel_types
        }
