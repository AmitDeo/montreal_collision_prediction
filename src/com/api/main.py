from fastapi import FastAPI

from com.api.routers import linear_regression


def main():
    app = FastAPI()
    app.include_router(linear_regression.router)
    return app


if __name__ == "__main__":
    main()
