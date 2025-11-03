"""
图神经网络风险检测MCP服务器
功能：数据集推理、模型信息查询、健康检查
"""
import os
import io
import base64
import shutil
import asyncio
from typing import Optional, Dict, Any, BinaryIO, List
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import uvicorn
import torch
import logging

from main import InferenceModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局推理模型实例
inference_model: Optional[InferenceModel] = None

# 创建MCP服务器
mcp = FastMCP("图神经网络风险检测服务器")


def get_inference_model() -> InferenceModel:
    """获取推理模型实例"""
    global inference_model
    if inference_model is None:
        raise RuntimeError("推理模型未初始化")
    if inference_model.model is None:
        raise RuntimeError("模型文件未正确加载，请检查模型文件路径是否正确")
    return inference_model


@mcp.tool(description="上传数据集ZIP文件并进行图神经网络推理预测")
async def predict_from_dataset(
    file_base64: str,
    filename: str = "dataset.zip"
) -> Dict[str, Any]:
    """
    上传数据集并进行推理
    
    Args:
        file_base64: Base64编码的ZIP文件内容
        filename: 文件名（默认dataset.zip）
        
    Returns:
        dict: 包含推理结果的字典
        
    示例：
        file_base64: "UEsDBBQAAAAIAC..."
        filename: "test_dataset.zip"
    """
    def _process():
        """同步处理函数，在线程中执行"""
        try:
            logger.info(f"开始处理数据集推理: {filename}")
            
            # 验证文件类型
            if not filename.endswith('.zip'):
                return {
                    'success': False,
                    'error': '文件必须是ZIP格式'
                }
            
            # 解码Base64文件内容
            try:
                file_bytes = base64.b64decode(file_base64)
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Base64解码失败: {str(e)}'
                }
            
            # 创建临时文件对象
            file_obj = io.BytesIO(file_bytes)
            
            # 获取推理模型
            model = get_inference_model()
            
            # 处理数据集
            dataset_path = model.process_uploaded_dataset(file_obj)
            logger.info(f"数据集处理完成: {dataset_path}")
            
            # 进行推理
            result = model.infer(dataset_path)
            logger.info("推理完成")
            
            return {
                'success': True,
                'result': result,
                'filename': filename
            }
            
        except Exception as e:
            logger.exception("推理过程出错")
            return {
                'success': False,
                'error': f"推理过程出错: {str(e)}"
            }
    
    # 在线程池中执行阻塞操作
    return await asyncio.to_thread(_process)


@mcp.tool(description="从指定路径加载数据集并进行推理")
async def predict_from_path(
    dataset_path: str
) -> Dict[str, Any]:
    """
    从指定路径加载数据集并进行推理
    
    Args:
        dataset_path: 数据集目录路径（包含meta.yaml等文件）
        
    Returns:
        dict: 包含推理结果的字典
    """
    def _process():
        """同步处理函数，在线程中执行"""
        try:
            logger.info(f"从路径加载数据集: {dataset_path}")
            
            # 验证路径存在
            if not os.path.exists(dataset_path):
                return {
                    'success': False,
                    'error': f'数据集路径不存在: {dataset_path}'
                }
            
            # 获取推理模型
            model = get_inference_model()
            
            # 进行推理
            result = model.infer(dataset_path)
            logger.info("推理完成")
            
            return {
                'success': True,
                'result': result,
                'dataset_path': dataset_path
            }
            
        except Exception as e:
            logger.exception("推理过程出错")
            return {
                'success': False,
                'error': f"推理过程出错: {str(e)}"
            }
    
    # 在线程池中执行阻塞操作
    return await asyncio.to_thread(_process)


@mcp.tool(description="获取当前加载的模型信息")
async def get_model_info() -> Dict[str, Any]:
    """
    获取模型信息
    
    Returns:
        dict: 包含模型配置和状态的字典
    """
    try:
        model = get_inference_model()
        
        return {
            'success': True,
            'model_path': model.model_path,
            'device': str(model.device),
            'model_loaded': model.model is not None,
            'model_type': 'GNN Risk Detection Model'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"获取模型信息失败: {str(e)}"
        }


@mcp.tool(description="检查服务健康状态")
async def health_check() -> Dict[str, Any]:
    """
    健康检查
    
    Returns:
        dict: 服务健康状态信息
    """
    try:
        model = get_inference_model()
        
        return {
            'status': 'healthy',
            'service': 'gnn-risk-detection-mcp-server',
            'model_loaded': model.model is not None,
            'device': str(model.device),
            'cuda_available': torch.cuda.is_available()
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """创建支持SSE传输和健康检查的Starlette应用"""
    sse = SseServerTransport("/messages/")
    
    async def handle_sse(request: Request):
        """处理SSE连接"""
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )
        # 返回空响应（连接已由SSE传输处理）
        from starlette.responses import Response
        return Response()
    
    async def health_endpoint(request: Request):
        """健康检查端点"""
        try:
            model = get_inference_model()
            return JSONResponse({
                "status": "healthy",
                "service": "gnn-risk-detection-mcp-server",
                "model_loaded": model.model is not None,
                "device": str(model.device)
            })
        except Exception as e:
            return JSONResponse(
                {"status": "unhealthy", "error": str(e)},
                status_code=503
            )
    
    async def root_info(request: Request):
        """根路径信息"""
        return JSONResponse({
            "service": "图神经网络风险检测MCP服务器",
            "version": "1.0.0",
            "description": "基于GNN的风险检测推理服务",
            "endpoints": {
                "root": "/",
                "health": "/health",
                "sse": "/sse",
                "messages": "/messages/"
            },
            "tools": [
                {
                    "name": "predict_from_dataset",
                    "description": "上传数据集ZIP文件并进行推理"
                },
                {
                    "name": "predict_from_path",
                    "description": "从指定路径加载数据集并进行推理"
                },
                {
                    "name": "get_model_info",
                    "description": "获取模型信息"
                },
                {
                    "name": "health_check",
                    "description": "检查服务健康状态"
                }
            ]
        })
    
    return Starlette(
        debug=debug,
        routes=[
            Route("/", endpoint=root_info, methods=["GET"]),
            Route("/health", endpoint=health_endpoint, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


def initialize_model(
    model_path: str = 'checkpoint/model.pt',
    device: Optional[str] = None,
    in_feats: int = 211,
    h_feats: int = 211,
    out_feats: int = 3
) -> None:
    """初始化推理模型"""
    global inference_model
    
    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"初始化模型: {model_path}")
    logger.info(f"使用设备: {device}")
    
    # 验证模型文件是否存在
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        logger.warning("将继续初始化，但模型将在文件可用时加载")
    
    # 创建推理模型实例
    inference_model = InferenceModel(model_path=model_path, device=device)
    
    # 如果模型文件存在，则加载模型
    if os.path.exists(model_path):
        inference_model.load_model(
            in_feats=in_feats,
            h_feats=h_feats,
            out_feats=out_feats
        )
        logger.info("✓ 模型加载成功")
    else:
        logger.warning("模型文件不存在，跳过加载")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='运行图神经网络风险检测MCP服务器'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='绑定的主机地址 (默认: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='监听的端口 (默认: 8000)'
    )
    parser.add_argument(
        '--model',
        default='checkpoint/model.pt',
        help='模型文件路径 (默认: checkpoint/model.pt)'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default=None,
        help='计算设备 (默认: 自动检测)'
    )
    parser.add_argument(
        '--in-feats',
        type=int,
        default=211,
        help='输入特征维度 (默认: 211)'
    )
    parser.add_argument(
        '--h-feats',
        type=int,
        default=211,
        help='隐藏层维度 (默认: 211)'
    )
    parser.add_argument(
        '--out-feats',
        type=int,
        default=3,
        help='输出类别数 (默认: 3)'
    )
    
    args = parser.parse_args()
    
    # 初始化推理模型
    initialize_model(
        model_path=args.model,
        device=args.device,
        in_feats=args.in_feats,
        h_feats=args.h_feats,
        out_feats=args.out_feats
    )
    
    # 获取MCP服务器实例
    mcp_server = mcp._mcp_server
    
    # 创建Starlette应用
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    # 打印启动信息
    print("=" * 60)
    print("图神经网络风险检测MCP服务器")
    print("=" * 60)
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"模型路径: {args.model}")
    print(f"计算设备: {args.device or '自动检测'}")
    print()
    print("可用端点:")
    print(f"  - GET  /          服务信息")
    print(f"  - GET  /health    健康检查")
    print(f"  - SSE  /sse       MCP连接")
    print(f"  - POST /messages/ MCP消息")
    print()
    print("MCP工具:")
    print("  - predict_from_dataset  上传ZIP数据集并推理")
    print("  - predict_from_path     从路径加载数据集并推理")
    print("  - get_model_info        获取模型信息")
    print("  - health_check          健康检查")
    print("=" * 60)
    
    # 运行服务器
    uvicorn.run(starlette_app, host=args.host, port=args.port)