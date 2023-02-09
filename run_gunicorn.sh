#!/bin/bash

<<EOF
run the app with gunicorn instead of directly calling uvicorn
EOF

gunicorn "application.main:app" -c application/config/gunicorn_config.py -k uvicorn.workers.UvicornWorker