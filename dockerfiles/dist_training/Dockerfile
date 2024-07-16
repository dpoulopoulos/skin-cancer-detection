FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.4-cuda12.4.0

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src src

WORKDIR /src

CMD [ "python", "main.py" ]
