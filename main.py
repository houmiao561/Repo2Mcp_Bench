
# 入口文件
# 先调用 mock 生成 server，部署 docker 中，在8000端口运行
# 然后调用 evaluate/call_mcp_server.py 进行测试端口是否调通
# 最后调用 evaluate/test_function.py 进行测试server的功能
# 生成功能保存在 result 中

import repo2mcpMock.mock
import evaluate.gnn_eval.call_mcp_server
import evaluate.gnn_eval.test_function
import evaluate.aml_eval.call_mcp_server
import evaluate.aml_eval.test_function
import subprocess
import time

def run_cmd(cmd):
    print(f"🧩 CMD运行: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 错误: {result.stderr.strip()}")
    else:
        print(f"✅ run_cmd结果： {result.stdout.strip()}\n")
    return result

def start_container(name, image, port=8000):
    run_cmd(f"docker rm -f {name} > /dev/null 2>&1 || true") # 删除可能存在的旧容器
    run_cmd(f"docker run -d --name {name} -p {port}:8000 {image}")
    print(f"✅ {name} 启动： {port}\n")
    time.sleep(5)  # 等待服务就绪

def stop_container(name):
    """停止容器"""
    run_cmd(f"docker stop {name}")
    print(f"✅ {name} 终止\n")

def test_gnn():
    repo2mcpMock.mock.convert_repo_to_mcp()
    print("\n🧩 检查连通性:")
    evaluate.gnn_eval.call_mcp_server.health_check()
    evaluate.gnn_eval.call_mcp_server.get_model_info()
    evaluate.gnn_eval.call_mcp_server.predict_from_path("123")
    evaluate.gnn_eval.call_mcp_server.predict_from_dataset("123")
    print("\n\n🧩 检查功能性:\n")
    evaluate.gnn_eval.test_function.test_function()

def test_aml():
    repo2mcpMock.mock.convert_repo_to_mcp()
    print("\n🧩 检查连通性:")
    evaluate.aml_eval.call_mcp_server.get_model_info()
    evaluate.aml_eval.call_mcp_server.caclulate_transaction_risk()
    evaluate.aml_eval.call_mcp_server.pridict_batch_customers()
    evaluate.aml_eval.call_mcp_server.predict_customer_risk()
    print("\n\n🧩 检查功能性:\n")
    evaluate.aml_eval.test_function.test_function()

def main():
    # Step 1: 启动并测试 GNN
    print("\n\n")
    print("\n\n")
    print("\n\n")
    print("🌟开始第一个测试GNN\n")
    start_container("gnn-mcp-service", "gnn-mcp-service:latest")
    try:
        test_gnn()
    finally:
        stop_container("gnn-mcp-service")

    print("\n\n")
    print("\n\n")
    print("\n\n")
    print("🌟开始第二个测试AML\n")

    # Step 2: 启动并测试 AML
    start_container("aml-mcp-service", "aml-mcp-service:latest")
    try:
        test_aml()
    finally:
        stop_container("aml-mcp-service")

if __name__ == "__main__":
    main()
