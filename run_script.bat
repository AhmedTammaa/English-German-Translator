docker build -t test .
docker run --rm --mount type=bind,source="C:\Users\Tammaa\Machine Translation\Machine_Translation",target=/docker -p 8888:8888 --gpus all test

sleep (10000)