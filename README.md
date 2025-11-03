# 🧪 repo2mcp-bench

    测试 Repo2Mcp框架 将 原始repo转换为mcpserver 的转换能力

## 📂 目录说明

    repo2mcp-bench/
    ├── Data/ # 各个Repo的原始源代码(原始考题) 与 转换后的Mcp(标准答案)
    ├── evaluate/ # 评测指标
    ├── repo2mcpMock/ # 待测评的框架，这里应该是实际的项目
    ├── results/ # 输出结果，包括生成的文件与综合评分
    ├── run_benchmark.py # 入口文件
    ├── requirements.txt # 依赖
    └── README.md

## 🚀 运行方法

    ```bash
    conda create -n repo2mcp python=3.11
    cd repo2mcp-bench
    pip install -r requirements.txt
    python run_benchmark.py
    ```
    输出结果保存在 results 文件夹中
    其中 {name}_candidate 是待测评框架生成的项目（这里先写死了）
    summary.json 是最终评分
