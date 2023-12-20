from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.router import router


def get_application() -> FastAPI:
    application = FastAPI(
        title="epf-flower-data-science",
        description="""Fast API""",
        version="1.0.0",
        redoc_url=None,
    )
    
    @router.get("/")
    def root():
        return RedirectResponse(url='/docs')

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application
