"""Start the FastAPI prediction service."""
import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "example_pipeline.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
