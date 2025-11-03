# 反洗钱（AML）风险预测 MCP 服务

基于机器学习的反洗钱风险预测服务，通过 Model Context Protocol (MCP) 提供 API 接口。

## 功能概述

本服务提供以下功能：

1. **单客户风险预测**：评估单个客户的反洗钱风险等级
2. **批量客户风险预测**：同时评估多个客户的风险状况
3. **交易风险评分**：计算单笔交易的风险评分
4. **模型信息查询**：获取当前加载模型的详细信息

## MCP 工具列表

| 工具名称 | 描述 | 参数 |
|---------|------|------|
| `predict_customer_risk` | 预测单个客户的反洗钱风险等级 | `customer_data`, `transactions`, `country_risk_mapping` |
| `predict_batch_customers` | 批量预测多个客户的反洗钱风险 | `customers_data`, `transactions_data`, `country_risk_mapping` |
| `calculate_transaction_risk` | 计算单笔交易的风险评分 | `transaction_id`, `amount`, `is_cash_transaction`, `is_cross_border`, `country_risk_score` |
| `get_model_info` | 获取当前加载的模型信息 | 无 |

## 本地开发运行

### 前提条件

- Python 3.10 或更高版本
- 安装所有依赖：`pip install -r requirements.txt`
- 确保 `models` 目录中有预训练的模型文件

### 运行服务

基本运行方式：

```bash
python app.py
```

指定主机和端口：

```bash
python app.py --host localhost --port 8001
```

指定模型路径：

```bash
python app.py --model /path/to/your/model.pkl
```

## SSE 端点访问说明

本服务使用 Server-Sent Events (SSE) 实现 MCP 通信协议。

- **SSE 连接端点**：`/sse`
- **消息处理端点**：`/messages/`
- **健康检查端点**：`/health`

## 容器化部署指南

### 使用 Docker

构建镜像：

```bash
docker build -t aml-mcp-service:latest .
```

运行容器：

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models aml-mcp-service:latest
```

### 使用 Docker Compose

启动服务：

```bash
docker-compose up -d
```

停止服务：

```bash
docker-compose down
```

## MCP 客户端连接配置示例

### Python 客户端

```python
from mcp.client import Client
from mcp.client.sse import SseClientTransport

# 创建 SSE 传输
transport = SseClientTransport(
    sse_url="http://localhost:8000/sse",
    post_url="http://localhost:8000/messages/"
)

# 创建 MCP 客户端
client = Client(transport)

# 连接到服务器
async with client.connect() as session:
    # 获取可用工具
    tools = await session.list_tools()
    print(f"可用工具: {tools}")
    
    # 调用预测工具
    result = await session.call_tool(
        "predict_customer_risk",
        customer_data={
            "customer_id": "CUST123",
            "name": "John Doe",
            "age": 45,
            "nationality": "United States",
            "occupation": "Business Owner",
            "account_opening_date": "2020-01-15",
            "pep_status": 0,
            "sanctions_match": 0,
            "address_change_count": 1
        },
        transactions=[
            {
                "transaction_id": "TXN001",
                "customer_id": "CUST123",
                "transaction_date": "2024-01-10 10:30:00",
                "amount": 5000.00,
                "transaction_type": "deposit",
                "is_cash_transaction": 1,
                "is_cross_border": 0,
                "country_code": "US"
            }
        ]
    )
    
    print(f"预测结果: {result}")
```

## 环境变量和配置说明

| 环境变量 | 描述 | 默认值 |
|---------|------|-------|
| `MODEL_PATH` | 模型文件路径 | `models/aml_model_random_forest.pkl` |
| `HOST` | 服务绑定的主机地址 | `0.0.0.0` |
| `PORT` | 服务监听的端口 | `8000` |
| `PYTHONUNBUFFERED` | Python 输出不缓冲 | `1` |

## 数据格式说明

### 客户数据格式

```json
{
    "customer_id": "CUST123",
    "name": "John Doe",
    "age": 45,
    "nationality": "United States",
    "occupation": "Business Owner",
    "account_opening_date": "2020-01-15",
    "pep_status": 0,
    "sanctions_match": 0,
    "address_change_count": 1
}
```

### 交易数据格式

```json
{
    "transaction_id": "TXN001",
    "customer_id": "CUST123",
    "transaction_date": "2024-01-10 10:30:00",
    "amount": 5000.00,
    "transaction_type": "deposit",
    "is_cash_transaction": 1,
    "is_cross_border": 0,
    "country_code": "US"
}
```

### 国家风险映射格式

```json
{
    "US": 0,  // 低风险
    "CN": 1,  // 中风险
    "AF": 2   // 高风险
}
```

## 错误处理

服务会在以下情况返回错误：

1. 模型未加载或加载失败
2. 输入数据格式不正确
3. 必要参数缺失
4. 服务内部错误

健康检查端点 `/health` 可用于监控服务状态。