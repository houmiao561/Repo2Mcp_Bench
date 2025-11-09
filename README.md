{
"mcpServers": {
"fetch": {
"type": "streamable_http",
"url": "https://mcp.api-inference.modelscope.net/4b3cc232674b4f/mcp"
}
}
}

curl http://localhost:3000/mcp \
 -H "Content-Type: application/json" \
 -d '{
"jsonrpc": "2.0",
"id": 1,
"method": "fetch",
"params": {
"url": "https://google.com"
}
}'
