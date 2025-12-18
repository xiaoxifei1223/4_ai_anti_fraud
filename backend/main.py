"""
FastAPI 主应用
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logger import logger
from app.db.database import init_db
from app.api import fraud_detection, health, users, rules, analytics
from app.services.model_service import model_service


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="基于机器学习和图神经网络的反欺诈检测系统",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境需要限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(
        health.router,
        prefix=settings.API_V1_PREFIX,
        tags=["健康检查"]
    )
    
    app.include_router(
        fraud_detection.router,
        prefix=settings.API_V1_PREFIX + "/fraud",
        tags=["欺诈检测"]
    )
    
    app.include_router(
        users.router,
        prefix=settings.API_V1_PREFIX + "/users",
        tags=["用户管理"]
    )
    
    app.include_router(
        rules.router,
        prefix=settings.API_V1_PREFIX + "/rules",
        tags=["规则管理"]
    )
    
    app.include_router(
        analytics.router,
        prefix=settings.API_V1_PREFIX,
        tags=["数据分析与监控"]
    )
    
    # 启动事件
    @app.on_event("startup")
    async def startup_event():
        """应用启动时执行"""
        logger.info(f"启动 {settings.PROJECT_NAME} v{settings.VERSION}")
        logger.info("初始化数据库...")
        init_db()
        logger.info("数据库初始化完成")
        
        # 加载 XGBoost 模型
        logger.info("加载 XGBoost 模型...")
        if model_service.load_model():
            logger.info("模型加载成功")
        else:
            logger.warning("模型加载失败，系统将使用规则引擎")
        
        logger.info(f"服务器运行在 http://{settings.HOST}:{settings.PORT}")
    
    # 关闭事件
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用关闭时执行"""
        logger.info("正在关闭服务...")
    
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
