# =============================================================================
#  API + CLI de ingesta.
#
#  Build multi-etapa: las dependencias se instalan en `builder` y se copia solo
#  el virtualenv resultante, dejando fuera compiladores y cachés de pip.
#
#  Incluye **Google Chrome estable**, no el Chromium empaquetado por Playwright:
#  medido contra el sitio real, el Chromium de Playwright es bloqueado con 403
#  por la protección anti-bot de bbva.com.co incluso con parches de stealth,
#  mientras que Chrome estable en headless pasa sin necesidad de Xvfb.
# =============================================================================

# --------------------------------------------------------------- builder ----
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip setuptools wheel && pip install .

# --------------------------------------------------------------- runtime ----
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    MODEL_CACHE_DIR=/app/models_cache

# Google Chrome estable desde el repositorio oficial + dependencias de sistema
# que Playwright necesita para controlarlo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg fonts-liberation libnss3 libatk1.0-0 \
        libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
        libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
        libcairo2 libatspi2.0-0 \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] \
        http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Chromium empaquetado como red de seguridad si Chrome fallara al lanzarse.
RUN playwright install chromium

# Usuario sin privilegios: el contenedor no necesita root en ejecución.
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/data/raw /app/data/clean /app/models_cache \
    && chown -R app:app /app /ms-playwright

WORKDIR /app
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "rag_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
