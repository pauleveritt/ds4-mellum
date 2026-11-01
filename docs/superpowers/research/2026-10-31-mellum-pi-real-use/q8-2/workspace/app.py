from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)