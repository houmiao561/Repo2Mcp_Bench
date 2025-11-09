import requests
import json

BASE_URL = "http://localhost:8000"

def health_check():
    """调用健康检查接口"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✅ Health check response:", response.json())
    except Exception as e:
        print("❌ Health check failed:", e)

def get_model_info():
    """调用 get_model_info 工具"""
    # MCP 服务通常通过 /messages/ 接收工具调用（类似 function calling）
    payload = {
        "tool": "get_model_info",
        "arguments": {}
    }
    try:
        response = requests.post(f"{BASE_URL}/messages/", json=payload)
        print("✅ get_model_info response:", response.json())
    except Exception as e:
        print("❌ get_model_info failed:", e)

def predict_from_path(dataset_path: str):
    """调用 predict_from_path 工具"""
    payload = {
        "tool": "predict_from_path",
        "arguments": {
            "path": dataset_path
        }
    }
    try:
        response = requests.post(f"{BASE_URL}/messages/", json=payload)
        print("✅ predict_from_path response:", response.json())
    except Exception as e:
        print("❌ predict_from_path failed:", e)

if __name__ == "__main__":
    print("🧪 Testing MCP Server at localhost:8000\n")

    # 1. 检查服务是否存活
    health_check()
    print()

    # 2. 获取模型信息
    get_model_info()
    print()

    # 3. （可选）示例：调用路径推理（请替换为你容器内存在的路径，或先用 predict_from_dataset 上传）
    predict_from_path("/app/data/sample.zip")