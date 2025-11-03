"""
反洗钱（AML）预测模块
从完整的训练系统中提取的轻量级预测服务

功能：
- 加载预训练的模型
- 单笔交易预测
- 批量交易预测
- 客户风险评估
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, List, Union, Optional, Generator


class AMLPredictor:
    """反洗钱预测器 - 轻量级版本，仅用于预测"""
    
    def __init__(self, model_path: str):
        """
        初始化预测器
        
        Args:
            model_path: 预训练模型文件路径 (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载预训练的模型"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            print(f"正在加载模型: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            print("[SUCCESS] 模型加载成功")
            
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {str(e)}")
            raise
    
    def _calculate_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个客户的聚合交易特征
        
        Args:
            transactions_df: 交易数据DataFrame，必须包含以下列：
                - customer_id: 客户ID
                - transaction_type: 交易类型
                - amount: 交易金额
                - is_cash_transaction: 是否现金交易（0/1）
                - is_cross_border: 是否跨境交易（0/1）
                - country_risk_score: 国家风险评分（0/1/2）
        
        Returns:
            包含聚合特征的DataFrame，以customer_id为索引
        """
        # 计算交易类型计数
        transaction_counts = transactions_df.groupby('customer_id')['transaction_type'].value_counts().unstack().fillna(0)
        
        # 确保所有交易类型都有列
        for txn_type in ['deposit', 'withdrawal', 'transfer', 'payment']:
            if txn_type not in transaction_counts.columns:
                transaction_counts[txn_type] = 0
        
        # 重命名列
        transaction_counts.columns = [f'{col}_count' for col in transaction_counts.columns]
        transaction_counts = transaction_counts.reset_index()
        
        # 计算聚合统计特征
        transaction_aggs = transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'std', 'max', 'count'],
            'is_cash_transaction': ['sum', 'mean'],
            'is_cross_border': ['sum', 'mean'],
            'country_risk_score': ['max', 'mean']
        })
        
        # 展平列名
        transaction_aggs.columns = ['_'.join(col).strip() for col in transaction_aggs.columns.values]
        
        # 重命名列使其更具描述性
        transaction_aggs = transaction_aggs.rename(columns={
            'amount_sum': 'total_transaction_amount',
            'amount_mean': 'avg_transaction_amount',
            'amount_std': 'std_transaction_amount',
            'amount_max': 'max_transaction_amount',
            'amount_count': 'transaction_count',
            'is_cash_transaction_sum': 'cash_transaction_count',
            'is_cash_transaction_mean': 'cash_transaction_ratio',
            'is_cross_border_sum': 'cross_border_count',
            'is_cross_border_mean': 'cross_border_ratio',
            'country_risk_score_max': 'max_country_risk',
            'country_risk_score_mean': 'avg_country_risk'
        })
        
        transaction_aggs = transaction_aggs.reset_index()
        
        # 计算现金交易金额
        cash_transactions = transactions_df[transactions_df['is_cash_transaction'] == 1]
        cash_by_customer = cash_transactions.groupby('customer_id')['amount'].sum().reset_index()
        cash_by_customer = cash_by_customer.rename(columns={'amount': 'total_cash_amount'})
        
        # 合并所有特征
        transaction_features = pd.merge(transaction_aggs, transaction_counts, on='customer_id', how='left')
        transaction_features = pd.merge(transaction_features, cash_by_customer, on='customer_id', how='left')
        
        # 填充缺失值并计算比率
        transaction_features['total_cash_amount'] = transaction_features['total_cash_amount'].fillna(0)
        transaction_features['cash_amount_ratio'] = (
            transaction_features['total_cash_amount'] / transaction_features['total_transaction_amount']
        )
        transaction_features['cash_amount_ratio'] = transaction_features['cash_amount_ratio'].fillna(0)
        
        return transaction_features
    
    def _fill_missing_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充没有交易的客户的交易特征值
        
        Args:
            df: 包含客户信息和交易特征的DataFrame
            
        Returns:
            填充后的DataFrame
        """
        # 交易相关的列（非客户基本信息列）
        transaction_cols = [col for col in df.columns if col not in 
                           ['customer_id', 'name', 'age', 'nationality', 
                            'occupation', 'account_opening_date', 'pep_status', 
                            'sanctions_match', 'address_change_count']]
        
        return df.fillna({col: 0 for col in transaction_cols})
    
    def predict_customer_risk(
        self, 
        customer_data: Dict, 
        transactions: List[Dict],
        country_risk_mapping: Optional[Dict[str, int]] = None
    ) -> Dict:
        """
        预测单个客户的风险等级
        
        Args:
            customer_data: 客户基本信息字典，包含：
                {
                    'customer_id': str,
                    'name': str,
                    'age': int,
                    'nationality': str,
                    'occupation': str,
                    'account_opening_date': str,
                    'pep_status': int (0/1),
                    'sanctions_match': int (0/1),
                    'address_change_count': int
                }
            
            transactions: 该客户的交易列表，每个交易包含：
                {
                    'transaction_id': str,
                    'customer_id': str,
                    'transaction_date': str,
                    'amount': float,
                    'transaction_type': str ('deposit', 'withdrawal', 'transfer', 'payment'),
                    'is_cash_transaction': int (0/1),
                    'is_cross_border': int (0/1),
                    'country_code': str
                }
            
            country_risk_mapping: 国家代码到风险评分的映射
                如: {'US': 0, 'CN': 1, 'AF': 2}  (0=low, 1=medium, 2=high)
                如果为None，默认所有国家风险为medium(1)
        
        Returns:
            预测结果字典：
                {
                    'customer_id': str,
                    'is_suspicious': int (0/1),
                    'suspicious_probability': float (0-1),
                    'risk_level': str ('low', 'medium', 'high'),
                    'prediction_time': str
                }
        """
        if not self.model_loaded:
            raise RuntimeError("模型未加载，无法进行预测")
        
        # 默认风险映射
        if country_risk_mapping is None:
            country_risk_mapping = {}
        
        # 准备客户数据
        customer_df = pd.DataFrame([customer_data])
        
        # 准备交易数据
        if len(transactions) == 0:
            # 没有交易的情况
            transactions_df = pd.DataFrame(columns=[
                'customer_id', 'transaction_type', 'amount', 
                'is_cash_transaction', 'is_cross_border', 'country_risk_score'
            ])
        else:
            transactions_df = pd.DataFrame(transactions)
            # 添加国家风险评分
            transactions_df['country_risk_score'] = transactions_df['country_code'].map(
                lambda x: country_risk_mapping.get(x, 1)  # 默认medium风险
            )
        
        # 计算交易特征
        if len(transactions_df) > 0:
            transaction_features = self._calculate_transaction_features(transactions_df)
            
            # 合并客户数据和交易特征
            customer_features = pd.merge(
                customer_df,
                transaction_features,
                on='customer_id',
                how='left'
            )
        else:
            customer_features = customer_df
        
        # 填充缺失值
        customer_features = self._fill_missing_transaction_features(customer_features)
        
        # 准备预测特征（移除不需要的列）
        X = customer_features.drop(['customer_id', 'name', 'account_opening_date'], axis=1)
        
        # 确保数据类型正确
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(float)
        
        # 进行预测
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        # 确定风险等级
        if probability < 0.3:
            risk_level = 'low'
        elif probability < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'customer_id': customer_data['customer_id'],
            'is_suspicious': int(prediction),
            'suspicious_probability': float(probability),
            'risk_level': risk_level,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def predict_batch_customers(
        self,
        customers_data: List[Dict],
        transactions_data: List[Dict],
        country_risk_mapping: Optional[Dict[str, int]] = None
    ) -> Generator[Dict, None, None]:
        """
        批量预测多个客户的风险（生成器模式，支持流式处理）
        
        Args:
            customers_data: 客户列表
            transactions_data: 所有交易列表
            country_risk_mapping: 国家风险映射
        
        Yields:
            每个客户的预测结果
        """
        # 将交易按客户分组
        transactions_df = pd.DataFrame(transactions_data)
        transactions_by_customer = {}
        
        if len(transactions_df) > 0:
            for customer_id, group in transactions_df.groupby('customer_id'):
                transactions_by_customer[customer_id] = group.to_dict('records')
        
        # 逐个预测
        for customer in customers_data:
            customer_id = customer['customer_id']
            customer_transactions = transactions_by_customer.get(customer_id, [])
            
            try:
                result = self.predict_customer_risk(
                    customer, 
                    customer_transactions, 
                    country_risk_mapping
                )
                yield result
            except Exception as e:
                # 记录错误但继续处理其他客户
                yield {
                    'customer_id': customer_id,
                    'error': str(e),
                    'is_suspicious': None,
                    'suspicious_probability': None
                }
    
    def calculate_transaction_risk_score(self, transaction: Dict) -> float:
        """
        计算单笔交易的风险评分（基于规则）
        
        Args:
            transaction: 交易数据字典
        
        Returns:
            风险评分（数值越大风险越高）
        """
        score = 0.0
        
        # 金额因素
        amount = transaction.get('amount', 0)
        score += amount / 1000  # 金额每1000增加1分
        
        # 现金交易
        if transaction.get('is_cash_transaction', 0) == 1:
            score += 3.0
        
        # 跨境交易
        if transaction.get('is_cross_border', 0) == 1:
            score += 2.0
        
        # 国家风险
        country_risk = transaction.get('country_risk_score', 1)
        score += country_risk
        
        return score
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型元信息字典
        """
        if not self.model_loaded:
            return {'loaded': False, 'error': '模型未加载'}
        
        try:
            model_type = type(self.model.named_steps['model']).__name__
            
            return {
                'loaded': True,
                'model_path': self.model_path,
                'model_type': model_type,
                'pipeline_steps': list(self.model.named_steps.keys())
            }
        except Exception as e:
            return {
                'loaded': True,
                'error': f'无法获取模型详细信息: {str(e)}'
            }


# 便捷函数
def load_predictor(model_path: str = 'models/aml_model_random_forest.pkl') -> AMLPredictor:
    """
    便捷函数：加载预测器
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        AMLPredictor实例
    """
    return AMLPredictor(model_path)


if __name__ == "__main__":
    """
    直接运行此文件进行反洗钱风险预测
    使用 example_data.py 中的示例数据
    """
    from example_data import (
        EXAMPLE_CUSTOMER,
        EXAMPLE_SUSPICIOUS_CUSTOMER,
        EXAMPLE_NORMAL_TRANSACTIONS,
        EXAMPLE_SUSPICIOUS_TRANSACTIONS,
        EXAMPLE_COUNTRY_RISK_MAPPING
    )
    
    print("\n" + "=" * 70)
    print(" 反洗钱（AML）风险预测系统")
    print("=" * 70)
    
    # 检查模型文件
    model_path = "models/aml_model_random_forest.pkl"
    if not os.path.exists(model_path):
        print(f"\n[错误] 模型文件不存在: {model_path}")
        print("请先运行主项目的 main.py 训练模型，然后将模型文件复制到此目录")
        sys.exit(1)
    
    try:
        # 加载预测器
        print("\n[步骤1] 加载预测模型...")
        predictor = load_predictor(model_path)
        
        # 显示模型信息
        model_info = predictor.get_model_info()
        print(f"  模型类型: {model_info.get('model_type', 'Unknown')}")
        
        # 预测1: 正常客户
        print("\n" + "-" * 70)
        print("[步骤2] 预测正常客户风险")
        print("-" * 70)
        print(f"客户信息: {EXAMPLE_CUSTOMER['name']}, {EXAMPLE_CUSTOMER['age']}岁")
        print(f"职业: {EXAMPLE_CUSTOMER['occupation']}, 国籍: {EXAMPLE_CUSTOMER['nationality']}")
        print(f"交易数量: {len(EXAMPLE_NORMAL_TRANSACTIONS)}笔")
        
        result1 = predictor.predict_customer_risk(
            customer_data=EXAMPLE_CUSTOMER,
            transactions=EXAMPLE_NORMAL_TRANSACTIONS,
            country_risk_mapping=EXAMPLE_COUNTRY_RISK_MAPPING
        )
        
        print(f"\n预测结果:")
        print(f"  客户ID: {result1['customer_id']}")
        print(f"  是否可疑: {'是 [!]' if result1['is_suspicious'] else '否 [OK]'}")
        print(f"  可疑概率: {result1['suspicious_probability']:.2%}")
        print(f"  风险等级: {result1['risk_level'].upper()}")
        print(f"  预测时间: {result1['prediction_time']}")
        
        # 预测2: 可疑客户
        print("\n" + "-" * 70)
        print("[步骤3] 预测可疑客户风险")
        print("-" * 70)
        print(f"客户信息: {EXAMPLE_SUSPICIOUS_CUSTOMER['name']}, {EXAMPLE_SUSPICIOUS_CUSTOMER['age']}岁")
        print(f"职业: {EXAMPLE_SUSPICIOUS_CUSTOMER['occupation']}, 国籍: {EXAMPLE_SUSPICIOUS_CUSTOMER['nationality']}")
        print(f"PEP身份: {'是' if EXAMPLE_SUSPICIOUS_CUSTOMER['pep_status'] else '否'}")
        print(f"地址变更: {EXAMPLE_SUSPICIOUS_CUSTOMER['address_change_count']}次")
        print(f"交易数量: {len(EXAMPLE_SUSPICIOUS_TRANSACTIONS)}笔")
        
        # 显示可疑交易特征
        total_amount = sum(t['amount'] for t in EXAMPLE_SUSPICIOUS_TRANSACTIONS)
        cash_count = sum(1 for t in EXAMPLE_SUSPICIOUS_TRANSACTIONS if t['is_cash_transaction'])
        cross_border_count = sum(1 for t in EXAMPLE_SUSPICIOUS_TRANSACTIONS if t['is_cross_border'])
        print(f"交易总额: ${total_amount:,.2f}")
        print(f"现金交易: {cash_count}笔, 跨境交易: {cross_border_count}笔")
        
        result2 = predictor.predict_customer_risk(
            customer_data=EXAMPLE_SUSPICIOUS_CUSTOMER,
            transactions=EXAMPLE_SUSPICIOUS_TRANSACTIONS,
            country_risk_mapping=EXAMPLE_COUNTRY_RISK_MAPPING
        )
        
        print(f"\n预测结果:")
        print(f"  客户ID: {result2['customer_id']}")
        print(f"  是否可疑: {'是 [!]' if result2['is_suspicious'] else '否 [OK]'}")
        print(f"  可疑概率: {result2['suspicious_probability']:.2%}")
        print(f"  风险等级: {result2['risk_level'].upper()}")
        print(f"  预测时间: {result2['prediction_time']}")
        
        # 总结
        print("\n" + "=" * 70)
        print(" 预测完成")
        print("=" * 70)
        print(f"\n共预测 2 个客户:")
        print(f"  - 正常客户: {result1['customer_id']} (风险等级: {result1['risk_level']})")
        print(f"  - 可疑客户: {result2['customer_id']} (风险等级: {result2['risk_level']})")
        print()
        
    except Exception as e:
        print(f"\n[错误] 预测失败: {str(e)}")
        import traceback
        traceback.print_exc()


