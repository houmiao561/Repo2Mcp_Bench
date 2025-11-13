
# 入口文件
# 先调用 mock 生成 server，部署 docker 中，在8000端口运行
# 然后调用 evaluate/call_mcp_server.py 进行测试端口是否调通
# 最后调用 evaluate/test_function.py 进行测试server的功能
# 生成功能保存在 result 中

import repo2mcpMock.mock
import evaluate.call_mcp_server
import evaluate.test_function

def main() :
    print("🧪 Testing MCP Server at localhost:8000\n")
    repo2mcpMock.mock.convert_repo_to_mcp()
    evaluate.call_mcp_server.health_check()
    evaluate.call_mcp_server.get_model_info()
    evaluate.call_mcp_server.predict_from_path("/app/data/sample.zip")
    print()
    print("接口连通性测试完毕，开始测试功能性:")
    print()
    evaluate.test_function.test_function()

if __name__ == "__main__" :
    main()
