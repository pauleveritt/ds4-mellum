from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def home():
    # Read base.html
    with open("templates/base.html") as f:
        base = f.read()
    # Hero content
    content = """
    <div class="hero">
        <h1>Come in. Sit down. Tell us about your human.</h1>
        <p>Welcome to AgentClinic. We're here to listen.</p>
    </div>
    """
    html = base.replace("{{content}}", content)
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run("__main__:app", reload=True)