# telltale — measurement harness for federated aggregation integrity.
# CPU-only image; the harness is deliberately small enough to run on one.
#
#   docker build -t telltale .
#   docker run --rm telltale                                   # runs the test suite
#   docker run --rm -v $PWD/results:/app/results telltale \
#       python scripts/run_probes.py --alpha 0.5              # one healthy run (downloads Fashion-MNIST)
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy scipy matplotlib pytest
COPY . .
CMD ["python", "-m", "pytest", "-q"]
