FROM karatayevm/dino-dragon-lambda:latest

COPY tflite_runtime-2.7.0-cp39-cp39-manylinux2014_x86_64.whl .

RUN pip install tflite_runtime-2.7.0-cp39-cp39-manylinux2014_x86_64.whl
RUN pip install pillow

COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]