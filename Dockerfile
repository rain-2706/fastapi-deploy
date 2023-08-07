FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /app

COPY ./app /app

RUN pip install fastapi uvicorn scikit-learn joblib

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app --reload"]

CMD ["uvicorn", "app.app:app", "8000"]