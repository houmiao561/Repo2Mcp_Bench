from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import uvicorn
from typing import List, Tuple, Dict, Any, Optional
import logging

from api.calculator import _calculate_linezolid_dose_impl
from api.linezolid_models import PatientData, DoseResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 MCP 服务器
mcp = FastMCP("利奈唑胺给药计算服务")

@mcp.tool(description="计算利奈唑胺的推荐剂量")
async def calculate_linezolid_dose(
    sex: int, 
    age: int, 
    height: int, 
    weight: int, 
    scr: float, 
    tb: float, 
    auc_range: List[float] = [160, 240]
) -> Dict[str, Any]:
    """
    计算利奈唑胺的推荐剂量
    
    Args:
        sex: 性别(1=男性, 0=女性)
        age: 年龄(岁)
        height: 身高(厘米)
        weight: 体重(千克)
        scr: 血清肌酐(μmol/L)
        tb: 总胆红素(μmol/L)
        auc_range: 目标AUC24h范围(min, max), 默认[160,240]
        
    Returns:
        dict: 包含计算结果的字典，包括体表面积、肾小球滤过率、推荐剂量、给药间隔、每日总剂量、预测AUC24h和目标AUC24h
    """
    try:
        logger.info(f"计算利奈唑胺剂量: 性别={sex}, 年龄={age}, 身高={height}, 体重={weight}, 肌酐={scr}, 胆红素={tb}")
        
        # 参数验证
        if sex not in [0, 1]:
            return {"error": "性别参数必须为0(女性)或1(男性)"}
        if age <= 0 or age > 100:
            return {"error": "年龄必须在1-100岁之间"}
        if height <= 0 or height > 200:
            return {"error": "身高必须在1-200厘米之间"}
        if weight <= 0 or weight > 200:
            return {"error": "体重必须在1-200千克之间"}
        if scr <= 0 or scr > 1000:
            return {"error": "血清肌酐必须在1-1000μmol/L之间"}
        if tb <= 0 or tb > 1000:
            return {"error": "总胆红素必须在1-1000μmol/L之间"}
        if len(auc_range) != 2 or auc_range[0] >= auc_range[1]:
            return {"error": "AUC范围必须是两个值，且第一个值小于第二个值"}
        
        # 调用内部实现函数
        result = _calculate_linezolid_dose_impl(
            sex=sex,
            age=age,
            height=height,
            weight=weight,
            scr=scr,
            tb=tb,
            auc_range=auc_range
        )
        
        logger.info(f"计算完成: 推荐剂量={result['dose']}mg, 间隔={result['interval']}h, AUC24={result['auc_24']}")
        return result
    
    except Exception as e:
        logger.error(f"计算过程出错: {str(e)}")
        return {"error": f"计算过程出错: {str(e)}"}


@mcp.tool(description="获取服务信息")
async def get_service_info() -> Dict[str, Any]:
    """
    获取服务信息
    
    Returns:
        dict: 包含服务信息的字典
    """
    return {
        "service": "利奈唑胺给药计算服务",
        "version": "1.0.0",
        "description": "基于群体药代动力学模型的利奈唑胺个体化给药计算服务",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages/"
        },
        "tools": [
            {
                "name": "calculate_linezolid_dose",
                "description": "计算利奈唑胺的推荐剂量"
            },
            {
                "name": "get_service_info",
                "description": "获取服务信息"
            }
        ]
    }


# 创建支持SSE的Starlette应用
def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """创建一个支持SSE传输的Starlette应用"""
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
    
    async def health_check(request: Request):
        """健康检查端点"""
        return JSONResponse({
            "status": "healthy",
            "service": "linezolid-mcp-service"
        })
    
    async def root_info(request: Request):
        """根路径信息"""
        return JSONResponse({
            "service": "利奈唑胺给药计算MCP服务",
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
            Mount("/messages", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行利奈唑胺给药计算MCP服务')
    parser.add_argument('--host', default='0.0.0.0', help='绑定的主机地址')
    parser.add_argument('--port', type=int, default=8000, help='监听的端口')
    args = parser.parse_args()
    
    # 获取MCP服务器
    mcp_server = mcp._mcp_server
    
    # 创建Starlette应用
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    # 打印启动信息
    print("=" * 60)
    print("利奈唑胺给药计算MCP服务")
    print("=" * 60)
    print(f"服务地址: http://{args.host}:{args.port}")
    print()
    print("可用端点:")
    print(f"  - GET  /          服务信息")
    print(f"  - GET  /health    健康检查")
    print(f"  - SSE  /sse       MCP连接")
    print(f"  - POST /messages/ MCP消息")
    print()
    print("MCP工具:")
    print("  - calculate_linezolid_dose  计算利奈唑胺的推荐剂量")
    print("  - get_service_info          获取服务信息")
    print("=" * 60)
    
    # 运行服务器
    uvicorn.run(starlette_app, host=args.host, port=args.port)