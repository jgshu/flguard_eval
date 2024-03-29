FROM ubuntu
RUN apt update && apt install -y python3 python3-pip python-is-python3 git

COPY requirements.txt /requirements.txt
RUN pip install jax && pip install -r requirements.txt

RUN python -c "import tenjin; import logging; logging.basicConfig(level=logging.INFO); tenjin.download('/app/data', 'mnist')"

COPY . /app
WORKDIR /app
ENTRYPOINT ["python3", "main.py"]