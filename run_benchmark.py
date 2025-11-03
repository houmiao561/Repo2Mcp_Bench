# run_benchmark.py
"""
主控脚本：运行整个 benchmark 流程
步骤：
1. 遍历 Data 目录下的所有 repo（aml_Repo, gnn_Repo, linezolid_Repo）
2. 对每个 repo，调用 repo2mcpMock.main 生成候选 MCP Server
3. 调用 evaluate.evaluator 对比候选 server 与 gold server 的功能
4. 输出结果到 results/summary.json
"""

import os
import json
from pathlib import Path
from repo2mcpMock.main import convert_repo_to_mcp  # 假设你的转换框架在这里
from evaluate.evaluator import evaluate_server  # 评测函数

# 定义要测试的 repo 列表
REPO_LIST = [
    "aml",
    "gnn",
    "linezolid"
]

def main():
    """主函数"""
    print("🚀 开始运行 Repo2MCP Benchmark...")

    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary = {}

    for repo_name in REPO_LIST:
        print(f"\n=== 正在评测 {repo_name} ===")

        # Step 1: 获取原始 repo 路径
        repo_path = Path("Data") / f"{repo_name}_Repo"
        if not repo_path.exists():
            print(f"❌ 错误：找不到原始 repo {repo_path}")
            continue

        # Step 2: 生成候选 server（输出到 results/{repo_name}_candidate）
        candidate_dir = results_dir / f"{repo_name}_candidate"
        candidate_dir.mkdir(exist_ok=True)

        print(f"  ➤ 生成候选 MCP Server 到 {candidate_dir}")
        convert_repo_to_mcp(str(repo_path), str(candidate_dir))

        # Step 3: 获取 gold server 路径（Data/{repo_name}_Mcp）
        gold_dir = Path("Data") / f"{repo_name}_Mcp"

        # Step 4: 评测候选 server vs gold server
        score = evaluate_server(
            candidate_dir=str(candidate_dir),
            gold_dir=str(gold_dir),
            repo_name=repo_name
        )

        # Step 5: 记录分数
        summary[repo_name] = score

        print(f"  ✅ {repo_name} 得分: {score}")

    # Step 6: 保存汇总结果
    summary_file = results_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 Benchmark 完成！结果已保存至 {summary_file}")

if __name__ == "__main__":
    main()