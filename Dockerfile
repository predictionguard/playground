FROM python:3.10

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY main.py ./
COPY .streamlit ./.streamlit

CMD ["streamlit", "run", "chat.py"]