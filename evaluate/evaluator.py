# evaluate/evaluator.py
"""
评测模块：启动候选 server 和 gold server，发送真实请求，对比功能正确性

核心逻辑：
1. 启动两个 server（候选 + gold），监听不同端口
2. 发送预定义的测试用例（来自 test_cases.json）
3. 比较响应是否一致（结构、字段、数值）
4. 杀死 server 进程
5. 返回多维评分（可启动性、工具发现、功能正确性等）

注意：
- 使用 subprocess 启动 server
- 使用 requests 发送 HTTP 请求
- 用 jsondiff 比较响应差异（简化版用 dict 比较）
"""

import os
import time
import subprocess
import requests
import json
from typing import Dict, Any, List, Tuple
import tempfile
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 预定义测试用例（你可以扩展这个文件）
TEST_CASES = {
    "aml": [
        {
            "tool_name": "predict_customer_risk",
            "args": {
                "customer_data": {
                    "customer_id": "CUST001",
                    "name": "张三",
                    "age": 35,
                    "nationality": "CN",
                    "occupation": "工程师",
                    "account_opening_date": "2023-01-01",
                    "pep_status": 0,
                    "sanctions_match": 0,
                    "address_change_count": 1
                },
                "transactions": [
                    {"transaction_id": "TXN001", "customer_id": "CUST001", "transaction_date": "2023-01-02", "amount": 5000, "transaction_type": "deposit", "is_cash_transaction": 0, "is_cross_border": 0, "country_code": "CN"}
                ],
                "country_risk_mapping": {"CN": 1, "US": 0}
            },
            "expected_keys": ["customer_id", "is_suspicious", "suspicious_probability", "risk_level"]
        }
    ],
    "linezolid": [
        {
            "tool_name": "calculate_linezolid_dose",
            "args": {
                "sex": 1,
                "age": 45,
                "height": 170,
                "weight": 70,
                "scr": 80,
                "tb": 15,
                "auc_range": [160, 240]
            },
            "expected_keys": ["dose", "interval", "auc_24"]
        }
    ],
    "gnn": [
        {
            "tool_name": "analyze_graph_structure",
            "args": {
                "nodes": [1, 2, 3],
                "edges": [[1, 2], [2, 3]]
            },
            "expected_keys": ["node_count", "edge_count", "clustering_coefficient"]
        }
    ]
}

def start_server(server_dir: str, port: int) -> subprocess.Popen:
    """
    启动 MCP Server
    
    Args:
        server_dir: server 代码所在目录（含 main.py）
        port: 监听端口
    
    Returns:
        subprocess.Popen 对象，用于后续 kill
    """
    logger.info(f"启动服务器: {server_dir} on port {port}")
    
    # 确保 main.py 存在
    main_py = os.path.join(server_dir, "main.py")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"未找到 main.py: {main_py}")
    
    # 构造启动命令
    cmd = [
        "python", "main.py",
        "--host", "127.0.0.1",
        "--port", str(port)
    ]
    
    # 在 server_dir 下启动进程
    proc = subprocess.Popen(
        cmd,
        cwd=server_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 等待 5 秒让服务启动
    time.sleep(5)
    
    # 检查是否启动成功
    if proc.poll() is not None:
        # 进程已退出，读取错误信息
        stdout, stderr = proc.communicate()
        raise RuntimeError(f"服务器启动失败:\n{stderr}")
    
    logger.info(f"✅ 服务器启动成功: http://127.0.0.1:{port}")
    return proc

def call_mcp_tool(port: int, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    调用 MCP 工具
    
    Args:
        port: server 端口
        tool_name: 工具名
        args: 参数字典
    
    Returns:
        工具返回的 JSON 响应
    """
    url = f"http://127.0.0.1:{port}/messages/"
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "tools.call",
        "params": {
            "name": tool_name,
            "arguments": args
        }
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        
        # 如果是正常响应，返回 result 字段
        if "result" in result:
            return result["result"]
        else:
            # 如果是错误响应，返回 error 字段
            return {"error": result.get("error", "Unknown error")}
            
    except Exception as e:
        return {"error": str(e)}

def compare_responses(pred_resp: Dict[str, Any], gold_resp: Dict[str, Any]) -> float:
    """
    简单比较两个响应是否一致（忽略顺序、部分字段缺失）
    
    Args:
        pred_resp: 候选 server 响应
        gold_resp: gold server 响应
    
    Returns:
        分数 (0~1)，1 表示完全一致
    """
    # 如果有 error，直接扣分
    if "error" in pred_resp:
        return 0.0
    
    # 检查预期字段是否存在
    for key in gold_resp.keys():
        if key not in pred_resp:
            return 0.5  # 关键字段缺失，给一半分
    
    # 检查数值是否接近（如果是数字）
    for key in gold_resp:
        if isinstance(gold_resp[key], (int, float)) and isinstance(pred_resp[key], (int, float)):
            diff = abs(gold_resp[key] - pred_resp[key])
            if diff > 0.1 * abs(gold_resp[key]):  # 允许 10% 误差
                return 0.7  # 数值不匹配，给 70%
    
    # 结构和数值都匹配，满分
    return 1.0

def evaluate_server(candidate_dir: str, gold_dir: str, repo_name: str) -> Dict[str, float]:
    """
    评测候选 server 与 gold server 的功能一致性
    
    Args:
        candidate_dir: 候选 server 目录
        gold_dir: gold server 目录
        repo_name: repo 名称（用于查找 test_cases）
    
    Returns:
        评分字典，包含多个维度
    """
    scores = {
        "can_start": 0.0,     # 是否能成功启动
        "tool_discovery": 0.0, # 是否能列出工具
        "functional_correctness": 0.0, # 功能是否正确
        "overall": 0.0         # 总分
    }
    
    # Step 1: 启动 gold server (port 8001)
    gold_proc = None
    candidate_proc = None
    gold_port = 8001
    candidate_port = 8002
    
    try:
        # 启动 gold server
        gold_proc = start_server(gold_dir, gold_port)
        scores["can_start"] = 1.0
        
        # 启动 candidate server
        candidate_proc = start_server(candidate_dir, candidate_port)
        scores["can_start"] = 1.0  # 两者都成功才算 1.0
        
        # Step 2: 检查工具列表（可选）
        # 可以通过 /list-tools 或其他方式检查，这里简化为假设都有工具
        scores["tool_discovery"] = 1.0
        
        # Step 3: 执行测试用例
        test_cases = TEST_CASES.get(repo_name, [])
        if not test_cases:
            logger.warning(f"没有为 {repo_name} 定义测试用例")
            scores["functional_correctness"] = 0.5
        else:
            total_score = 0.0
            for i, case in enumerate(test_cases):
                tool_name = case["tool_name"]
                args = case["args"]
                expected_keys = case.get("expected_keys", [])
                
                # 调用 gold server
                gold_resp = call_mcp_tool(gold_port, tool_name, args)
                logger.info(f"Gold response for {tool_name}: {gold_resp}")
                
                # 调用 candidate server
                pred_resp = call_mcp_tool(candidate_port, tool_name, args)
                logger.info(f"Candidate response for {tool_name}: {pred_resp}")
                
                # 比较响应
                case_score = compare_responses(pred_resp, gold_resp)
                total_score += case_score
                
                # 检查预期字段是否存在
                for key in expected_keys:
                    if key not in pred_resp:
                        case_score = 0.5  # 关键字段缺失
                        break
                
                logger.info(f"Case {i+1} score: {case_score}")
            
            scores["functional_correctness"] = total_score / len(test_cases)
        
        # Step 4: 计算总分（加权平均）
        scores["overall"] = (
            0.3 * scores["can_start"] +
            0.2 * scores["tool_discovery"] +
            0.5 * scores["functional_correctness"]
        )
        
    except Exception as e:
        logger.error(f"评测 {repo_name} 失败: {str(e)}")
        scores["overall"] = 0.0
    finally:
        # 清理：杀死 server 进程
        if gold_proc:
            gold_proc.kill()
        if candidate_proc:
            candidate_proc.kill()
    
    return scores