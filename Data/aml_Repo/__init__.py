"""
AML 预测器模块

轻量级的反洗钱预测服务，从完整训练系统中提取
"""

from .predictor import AMLPredictor, load_predictor

__version__ = "1.0.0"
__all__ = ["AMLPredictor", "load_predictor"]

