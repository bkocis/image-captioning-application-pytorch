# application/config/gunicorn_config.py

import multiprocessing

# Binding
bind = "0.0.0.0:8081"  # Change the port if needed

# Worker Options
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1024

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Timeout
timeout = 120

# SSL Options (if needed)
# keyfile = "path/to/keyfile"
# certfile = "path/to/certfile"

# Process Naming
proc_name = "image_captioning_app"

# Server Mechanics
daemon = False
reload = True  # Enable auto-reload during development

# Header Settings
forwarded_allow_ips = "*"

# Worker Specific Settings
worker_tmp_dir = "/dev/shm"  # Improves performance on Linux systems
threads = 4

# Additional Uvicorn specific configs
worker_kwargs = {
    "proxy_headers": True,
    "forwarded_allow_ips": "*",
    "timeout_keep_alive": 30,
}