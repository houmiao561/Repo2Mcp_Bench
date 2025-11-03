"""
反洗钱（AML）预测MCP服务器
基于FastMCP + Starlette实现的MCP服务，提供反洗钱风险预测功能

功能：
- 单客户风险预测
- 批量客户风险预测
- 交易风险评分计算
- 模型信息查询
"""

import os
import argparse
from typing import Dict, List, Optional, Union, Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import uvicorn

from predictor import AMLPredictor, load_predictor
from example_data import EXAMPLE_COUNTRY_RISK_MAPPING

# 创建 MCP 服务器
mcp = FastMCP("反洗钱风险预测服务器")

# 全局预测器实例
predictor: Optional[AMLPredictor] = None

def get_predictor() -> AMLPredictor:
    """获取预测器实例"""
    global predictor
    if predictor is None:
        raise RuntimeError("预测器未初始化")
    return predictor

@mcp.tool(description="预测单个客户的反洗钱风险等级")
async def predict_customer_risk(
    customer_data: Dict[str, Any],
    transactions: List[Dict[str, Any]],
    country_risk_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    预测单个客户的反洗钱风险等级
    
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
    pred = get_predictor()
    return pred.predict_customer_risk(customer_data, transactions, country_risk_mapping)

@mcp.tool(description="批量预测多个客户的反洗钱风险")
async def predict_batch_customers(
    customers_data: List[Dict[str, Any]],
    transactions_data: List[Dict[str, Any]],
    country_risk_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    批量预测多个客户的反洗钱风险
    
    Args:
        customers_data: 客户列表，每个客户包含与predict_customer_risk相同的字段
        transactions_data: 所有交易列表
        country_risk_mapping: 国家风险映射
    
    Returns:
        包含所有客户预测结果的字典：
            {
                'total_customers': int,
                'results': List[Dict],
                'summary': {
                    'suspicious_count': int,
                    'suspicious_rate': str
                }
            }
    """
    pred = get_predictor()
    
    if country_risk_mapping is None:
        country_risk_mapping = EXAMPLE_COUNTRY_RISK_MAPPING
    
    results = list(pred.predict_batch_customers(customers_data, transactions_data, country_risk_mapping))
    suspicious_count = sum(1 for r in results if r.get('is_suspicious') == 1)
    
    return {
        'total_customers': len(customers_data),
        'results': results,
        'summary': {
            'suspicious_count': suspicious_count,
            'suspicious_rate': f"{(suspicious_count / len(customers_data) * 100):.2f}%" if customers_data else "0%"
        }
    }

@mcp.tool(description="计算单笔交易的风险评分")
async def calculate_transaction_risk(
    transaction_id: str,
    amount: float,
    is_cash_transaction: int,
    is_cross_border: int,
    country_risk_score: int = 1
) -> Dict[str, Any]:
    """
    计算单笔交易的风险评分
    
    Args:
        transaction_id: 交易ID
        amount: 交易金额
        is_cash_transaction: 是否现金交易 (0/1)
        is_cross_border: 是否跨境交易 (0/1)
        country_risk_score: 国家风险评分 (0=低风险, 1=中风险, 2=高风险)
    
    Returns:
        风险评分结果：
            {
                'transaction_id': str,
                'risk_score': float,
                'risk_level': str ('low', 'medium', 'high'),
                'risk_factors': List[str]
            }
    """
    pred = get_predictor()
    transaction = {
        'transaction_id': transaction_id, 
        'amount': amount,
        'is_cash_transaction': is_cash_transaction,
        'is_cross_border': is_cross_border,
        'country_risk_score': country_risk_score
    }
    score = pred.calculate_transaction_risk_score(transaction)
    risk_level = 'high' if score >= 10 else ('medium' if score >= 5 else 'low')
    
    risk_factors = []
    if amount > 10000:
        risk_factors.append(f"大额交易 (${amount:,.2f})")
    if is_cash_transaction:
        risk_factors.append("现金交易")
    if is_cross_border:
        risk_factors.append("跨境交易")
    if country_risk_score >= 2:
        risk_factors.append("高风险国家/地区")
    
    return {
        'transaction_id': transaction_id,
        'risk_score': float(score),
        'risk_level': risk_level,
        'risk_factors': risk_factors if risk_factors else ["无明显风险因素"]
    }

@mcp.tool(description="获取当前加载的模型信息")
async def get_model_info() -> Dict[str, Any]:
    """
    获取当前加载的模型信息
    
    Returns:
        模型元信息字典：
            {
                'loaded': bool,
                'model_path': str,
                'model_type': str,
                'pipeline_steps': List[str]
            }
    """
    pred = get_predictor()
    return pred.get_model_info()

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """创建一个支持SSE传输的Starlette应用"""
    sse = SseServerTransport("/messages/")
    
    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
    
    async def health_check(request: Request) -> JSONResponse:
        """健康检查端点"""
        try:
            pred = get_predictor()
            model_info = pred.get_model_info()
            return JSONResponse({
                "status": "healthy",
                "service": "aml-predictor-mcp-server",
                "model_loaded": model_info.get('loaded', False)
            })
        except Exception as e:
            return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)
    
    async def root_info(request: Request) -> JSONResponse:
        """根路径信息"""
        return JSONResponse({
            "service": "反洗钱风险预测MCP服务器",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "sse": "/sse",
                "messages": "/messages/"
            }
        })
    
    return Starlette(
        debug=debug,
        routes=[
            Route("/", endpoint=root_info, methods=["GET"]),
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

def initialize_predictor(model_path: str = "models/aml_model_random_forest.pkl") -> None:
    """初始化预测器"""
    global predictor
    
    if not os.path.exists(model_path):
        print(f"[警告] 模型文件不存在: {model_path}")
        print("将在首次请求时尝试加载模型")
    
    try:
        predictor = load_predictor(model_path)
        print(f"[成功] 模型加载完成: {model_path}")
    except Exception as e:
        print(f"[错误] 模型加载失败: {str(e)}")
        print("将在首次请求时尝试加载模型")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行反洗钱预测MCP服务器')
    parser.add_argument('--host', default='0.0.0.0', help='绑定的主机地址')
    parser.add_argument('--port', type=int, default=8000, help='监听的端口')
    parser.add_argument('--model', default='models/aml_model_random_forest.pkl', help='模型文件路径')
    args = parser.parse_args()
    
    # 初始化预测器
    initialize_predictor(args.model)
    
    # 创建并运行服务
    mcp_server = mcp._mcp_server
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    print("=" * 60)
    print(" 反洗钱风险预测MCP服务器")
    print("=" * 60)
    print(f"服务已启动: http://{args.host}:{args.port}")
    print(f"  - GET  /health   健康检查")
    print(f"  - SSE  /sse      MCP连接")
    print(f"  - POST /messages/ MCP消息")
    print("=" * 60)
    
    uvicorn.run(starlette_app, host=args.host, port=args.port)