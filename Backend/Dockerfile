FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install torchvision==0.17.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html

#RUN pip install --no-cache-dir jax==0.4.34 jaxlib==0.4.34

#RUN pip install --no-cache-dir jax==0.4.34 \
    #jaxlib==0.4.34+cpu \
    #-f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

