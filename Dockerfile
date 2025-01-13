FROM python:3.11-slim

# Install all dependencies with poetry
RUN pip install poetry==1.8.5
WORKDIR /app
COPY ["pyproject.toml", "poetry.lock", "./"]
RUN poetry install --without notebook --no-root

# Copy Flask script
COPY ["predict.py", "model.bin", "./"]
EXPOSE 9696

# Run it with Gunicorn
ENTRYPOINT ["poetry", "run", "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
