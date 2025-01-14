# FROM python:3.11-slim
FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy lambda function
COPY ["lambda_function.py", "model.bin", "./"]

# Run lambda function
CMD [ "lambda_function.lambda_handler" ]
