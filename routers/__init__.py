from routers.admin import router as admin_router
from routers.ask import router as ask_router
from routers.health import router as health_router
from routers.ingest import router as ingest_router

routes = [health_router, ingest_router, ask_router, admin_router]
