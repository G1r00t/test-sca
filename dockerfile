FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y chromium-browser=78.0.3904.108-0ubuntu0.18.04.1 || true

CMD ["chromium-browser", "--version"]
