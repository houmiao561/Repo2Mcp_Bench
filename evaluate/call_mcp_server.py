# 用来测试接口能不能跑通

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
    payload = {
        "method": "tools/call",
        "params": {
            "name": "get_model_info",
            "arguments": {},
            "_meta": {
                "progressToken": 2
            }
        }
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(f"{BASE_URL}/messages", json=payload, headers=headers)
        print("✅ get_model_info response:", response)
    except Exception as e:
        print("❌ get_model_info failed: 这里没有搞定session_id，", e)


def predict_from_path(dataset_path: str):
    """调用 predict_from_path 工具"""
    payload = {
        "method": "tools/call",
        "params": {
            "name": "predict_from_path",
            "arguments": {
                "dataset_path": ""
            },
            "_meta": {
                "progressToken": 3
            }
        }
        
    }
    try:
        response = requests.post(f"{BASE_URL}/messages/", json=payload)
        print("✅ predict_from_path response:", response.json())
    except Exception as e:
        print("❌ predict_from_path failed:", e)
