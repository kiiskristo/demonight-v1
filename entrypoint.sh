#!/bin/sh
PORT=${PORT:-8080}  # Remove quotes, PORT is provided by Railway
exec uvicorn src.howdoyoufindme.main:app --host 0.0.0.0 --port $PORT