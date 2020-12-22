FROM python:3.9.1-buster

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

EXPOSE 5000

ENTRYPOINT [ "python" ] 
CMD [ "main.py" ] 