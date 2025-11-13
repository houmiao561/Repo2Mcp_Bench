# 在接口跑通的情况下，用来测试server的功能
import json
import base64
import requests
import sseclient

def test_function ():
    print("""🚀这里开始测试aml功能\n测试用例在哪里？""") 

    resp = requests.post(
        "http://localhost:8000/messages/",
        headers={"Content-Type": "application/json"}
    )

    if resp.status_code != 200:
        print("❌ test_function 调用失败:", resp.status_code)
        print()

    # 4. 从 SSE 流读取响应（/sse）
    # sse = sseclient.SSEClient("http://localhost:8000/sse")


