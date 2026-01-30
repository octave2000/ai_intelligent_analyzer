import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port_raw = os.getenv("PORT", "8000")
    try:
        port = int(port_raw)
    except ValueError:
        port = 8000
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
