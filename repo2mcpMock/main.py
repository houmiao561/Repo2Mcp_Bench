# repo2mcpMock/main.py
"""
模拟 repo2mcp 框架：将原始 repo 转换为 MCP Server
当前实现：直接复制 gold server（用于快速跑通流程）
未来替换为真实转换逻辑
"""

import shutil
import os

def convert_repo_to_mcp(repo_path: str, output_dir: str):
    """
    将原始 repo 转换为 MCP Server
    
    Args:
        repo_path: 原始 repo 路径
        output_dir: 输出目录
    """
    # 临时：直接复制对应 gold server
    # 例如：如果 repo_path 是 "Data/aml_Repo"，则复制 "Data/aml_Mcp" 到 output_dir
    repo_name = os.path.basename(repo_path).replace("_Repo", "")
    gold_dir = os.path.join("Data", f"{repo_name}_Mcp")
    
    if not os.path.exists(gold_dir):
        raise FileNotFoundError(f"找不到 gold server: {gold_dir}")
    
    # 复制整个目录
    shutil.copytree(gold_dir, output_dir, dirs_exist_ok=True)
    print(f"✅ 已复制 {gold_dir} 到 {output_dir}")