# 利奈唑胺给药计算 MCP 服务

基于群体药代动力学模型的利奈唑胺个体化给药计算服务，通过 Model Context Protocol (MCP) 提供 API 接口。

## 功能描述

本服务提供以下功能：

1. **利奈唑胺剂量计算**：根据患者的性别、年龄、身高、体重、血清肌酐和总胆红素等参数，计算个体化的利奈唑胺推荐剂量。
2. **服务信息查询**：获取服务的基本信息和可用工具列表。

### MCP 工具列表

| 工具名称 | 描述 | 参数 |
|---------|------|------|
| `calculate_linezolid_dose` | 计算利奈唑胺的推荐剂量 | `sex`, `age`, `height`, `weight`, `scr`, `tb`, `auc_range` |
| `get_service_info` | 获取服务信息 | 无 |

## 本地开发运行指南

### 环境准备

确保已安装 Python 3.10 或更高版本，并安装所需依赖：

```bash
pip install -r requirements.txt
```

### 运行服务

默认配置运行：

```bash
python app.py
```

自定义主机和端口：

```bash
python app.py --host localhost --port 8001
```

## SSE 端点访问说明

MCP 服务使用 Server-Sent Events (SSE) 进行通信，提供以下端点：

- `/sse` - SSE 连接端点
- `/messages/` - 消息处理挂载点

客户端可以通过连接 `/sse` 端点建立 SSE 连接，然后通过 `/messages/` 端点发送和接收消息。

## 容器化部署指南

### 使用 Docker 构建和运行

构建镜像：

```bash
docker build -t linezolid-mcp-service:latest .
```

运行容器：

```bash
docker run -p 8000:8000 linezolid-mcp-service:latest
```

### 使用 Docker Compose 部署

一键部署服务：

```bash
docker-compose up -d
```

停止服务：

```bash
docker-compose down
```

## MCP 客户端连接配置示例

### Python 客户端示例

```python
from mcp.client import Client
from mcp.client.sse import SseClientTransport

# 创建 SSE 传输
transport = SseClientTransport("http://localhost:8000/sse", "http://localhost:8000/messages/")

# 创建 MCP 客户端
client = Client(transport)

# 连接到服务器
client.connect()

# 调用计算利奈唑胺剂量的工具
result = client.call_tool(
    "calculate_linezolid_dose",
    sex=1,  # 男性
    age=45,
    height=170,
    weight=70,
    scr=80.0,
    tb=15.0,
    auc_range=[160, 240]
)

print(result)

# 断开连接
client.disconnect()
```

### JavaScript 客户端示例

```javascript
import { Client } from '@mcp/client';
import { SseClientTransport } from '@mcp/client/sse';

async function main() {
  // 创建 SSE 传输
  const transport = new SseClientTransport(
    'http://localhost:8000/sse',
    'http://localhost:8000/messages/'
  );

  // 创建 MCP 客户端
  const client = new Client(transport);

  // 连接到服务器
  await client.connect();

  try {
    // 调用计算利奈唑胺剂量的工具
    const result = await client.callTool(
      'calculate_linezolid_dose',
      {
        sex: 1,  // 男性
        age: 45,
        height: 170,
        weight: 70,
        scr: 80.0,
        tb: 15.0,
        auc_range: [160, 240]
      }
    );

    console.log(result);
  } finally {
    // 断开连接
    await client.disconnect();
  }
}

main().catch(console.error);
```

## 环境变量和配置说明

服务支持以下环境变量配置：

| 环境变量 | 描述 | 默认值 |
|---------|------|-------|
| `HOST` | 绑定的主机地址 | `0.0.0.0` |
| `PORT` | 监听的端口 | `8000` |

## 健康检查

服务提供了健康检查端点：

- `/health` - 返回服务健康状态

可以通过以下命令检查服务健康状态：

```bash
curl http://localhost:8000/health
```

## 许可证

[MIT License](LICENSE)