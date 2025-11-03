# 图神经网络风险检测MCP服务器 - Docker镜像
FROM python:3.10-slim

# 配置阿里云APT镜像源（自动适配Debian版本）
RUN . /etc/os-release && \
    echo "deb https://mirrors.aliyun.com/debian/ $VERSION_CODENAME main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ $VERSION_CODENAME-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security $VERSION_CODENAME-security main contrib non-free" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DGL_GRAPHBOLT_USE_CPP=0

# 复制依赖文件
COPY requirements.txt .

# 设置 pip 使用清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 先删除 requirements.txt 中的 dgl（如果有的话）
RUN sed -i '/^dgl/d' requirements.txt

# 安装其他 Python 依赖（使用清华镜像）
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 单独安装 DGL（使用清华镜像）
RUN pip install dgl==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY . .

# 创建checkpoint目录（用于存放模型文件）
RUN mkdir -p checkpoint

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# 设置默认环境变量
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=checkpoint/model.pt \
    IN_FEATS=211 \
    H_FEATS=211 \
    OUT_FEATS=3

# 启动命令
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]