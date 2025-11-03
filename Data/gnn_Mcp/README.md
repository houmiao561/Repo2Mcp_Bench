# 图神经网络风险检测 MCP 服务

基于图神经网络（GNN）的风险检测服务，使用 Model Context Protocol (MCP) 提供 API 接口。

## 功能概述

本服务使用图神经网络模型对数据进行风险检测分析，提供以下功能：

1. 上传数据集 ZIP 文件并进行推理
2. 从指定路径加载数据集并进行推理
3. 获取模型信息
4. 健康状态检查

## MCP 工具列表

| 工具名称 | 描述 | 参数 |
|---------|------|------|
| `predict_from_dataset` | 上传数据集ZIP文件并进行图神经网络推理预测 | `file_base64`: Base64编码的ZIP文件内容<br>`filename`: 文件名（默认dataset.zip） |
| `predict_from_path` | 从指定路径加载数据集并进行推理 | `dataset_path`: 数据集目录路径 |
| `get_model_info` | 获取当前加载的模型信息 | 无 |
| `health_check` | 检查服务健康状态 | 无 |

## 本地开发运行指南

### 环境准备

确保已安装 Python 3.10 及以上版本，并安装所需依赖：

```bash
pip install -r requirements.txt
```

### 运行服务

基本运行方式：

```bash
python app.py
```

指定主机和端口：

```bash
python app.py --host localhost --port 8001
```

其他可选参数：

```bash
python app.py --model checkpoint/model.pt --device cpu --in-feats 211 --h-feats 211 --out-feats 3
```

## SSE 端点访问说明

服务提供以下 HTTP 端点：

- `GET /`: 服务信息
- `GET /health`: 健康检查
- `GET /sse`: SSE 连接端点（用于 MCP 客户端连接）
- `POST /messages/`: MCP 消息处理端点

MCP 客户端应通过 `/sse` 端点建立 SSE 连接，并通过 `/messages/` 端点发送消息。

## 容器化部署指南

### 使用 Docker Compose 部署

1. 确保已安装 Docker 和 Docker Compose
2. 在项目根目录下运行：

```bash
docker-compose up -d
```

这将构建并启动 MCP 服务容器，服务将在 `http://localhost:8000` 上可用。

### 手动构建 Docker 镜像

如果需要手动构建 Docker 镜像：

```bash
docker build -t gnn-mcp-service:latest .
```

运行容器：

```bash
docker run -p 8000:8000 -v $(pwd)/checkpoint:/app/checkpoint gnn-mcp-service:latest
```

## MCP 客户端连接配置示例

使用 Python MCP 客户端连接服务：

```python
from mcp.client import Client
from mcp.client.sse import SseClientTransport

# 创建 SSE 传输
transport = SseClientTransport("http://localhost:8000/sse", "http://localhost:8000/messages/")

# 创建 MCP 客户端
client = Client(transport)

# 连接到服务器
client.connect()

# 获取可用工具
tools = client.list_tools()
print(f"可用工具: {tools}")

# 调用工具示例 - 获取模型信息
model_info = client.invoke_tool("get_model_info")
print(f"模型信息: {model_info}")

# 调用工具示例 - 从路径加载数据集并推理
result = client.invoke_tool("predict_from_path", {"dataset_path": "/path/to/dataset"})
print(f"推理结果: {result}")

# 断开连接
client.disconnect()
```

## 环境变量和配置说明

服务支持以下环境变量配置：

| 环境变量 | 描述 | 默认值 |
|---------|------|-------|
| `HOST` | 绑定的主机地址 | `0.0.0.0` |
| `PORT` | 监听的端口 | `8000` |
| `MODEL_PATH` | 模型文件路径 | `checkpoint/model.pt` |
| `IN_FEATS` | 输入特征维度 | `211` |
| `H_FEATS` | 隐藏层维度 | `211` |
| `OUT_FEATS` | 输出类别数 | `3` |

这些环境变量可以在 Docker Compose 配置中设置，或者在运行容器时通过 `-e` 参数指定。

## 数据集格式要求

上传的数据集 ZIP 文件应包含以下内容：

- `meta.yaml`: 数据集元数据文件
- 其他必要的数据文件（与训练数据格式相同）

数据集将使用 DGL 的 CSVDataset 加载，请确保数据格式符合要求。