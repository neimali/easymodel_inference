# 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器
COPY . /app

# 安装 Poetry
RUN pip install poetry

# 安装依赖
RUN poetry config virtualenvs.create false && poetry install --no-dev

# 暴露 Flask 端口
EXPOSE 5000

# 运行 Flask 应用
CMD ["python", "app/main.py"]
