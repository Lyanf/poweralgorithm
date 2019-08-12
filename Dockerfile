FROM python:3.6
WORKDIR /app
COPY oriCode.py .
COPY web.py .
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
EXPOSE 5000
ENV SHARED_ROOT /data
CMD ["python","web.py"]
