# 在接口跑通的情况下，用来测试server的功能
import json
import base64
import requests
# import sseclient

def test_function ():
    print("""这里开始测试GNN功能""")
    with open("/Users/houmiao/Desktop/Repo2Mcp_Bench/Data/gnn_datasets.zip", "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode()
        # print(f"zip_b64: {zip_b64}") 
    call_msg = {
        "jsonrpc": "2.0",
        "id": "bench_run_001",
        "method": "call",
        "params": {
            "name": "predict_from_dataset",
            "arguments": {
                "file_base64": zip_b64,
                "filename": "gnn_datasets.zip"
            }
        }
    }
    resp = requests.post(
        "http://localhost:8000/messages/",
        json=call_msg,
        headers={"Content-Type": "application/json"}
    )

    if resp.status_code != 200:
        print("❌ 调用失败:", resp.status_code)
        raise Exception(f"调用失败: {resp.text}")

    # 4. 从 SSE 流读取响应（/sse）
    # client = sseclient.SSEClient("http://localhost:8000/sse")
    # for event in client.events():
    #     if event.event == "message":
    #         data = json.loads(event.data)
    #         # 检查是否是你的请求响应
    #         if data.get("id") == "bench_run_001":
    #             print("✅ 推理结果:")
    #             print(json.dumps(data["result"], indent=2, ensure_ascii=False))
    #             break






