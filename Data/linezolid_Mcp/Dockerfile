FROM python:3.10-slim

# 配置阿里云APT镜像源（自动适配Debian版本）
RUN . /etc/os-release && \
    echo "deb https://mirrors.aliyun.com/debian/ $VERSION_CODENAME main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ $VERSION_CODENAME-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security $VERSION_CODENAME-security main contrib non-free" >> /etc/apt/sources.list

# 安装系统依赖（如有需要）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖（使用清华源）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# 启动命令
CMD ["python", "app.py"]