import gradio as gr
from fastapi import FastAPI, Response
import uvicorn
import threading
import time
from modules.async_worker import AsyncTask, async_tasks, worker
import shared

# 1. Locate the Python function or method that is responsible for generating the image
# The image generation is handled by the `worker` function in `modules/async_worker.py`.
# It processes tasks from the `async_tasks` queue.

# Create a FastAPI app
app = FastAPI()

# 2. Create a separate, new Python file (e.g., api_wrapper.py) that:
#    • Imports and uses that exact image generation function.
#    • Accepts only a "prompt" argument from the caller.
#    • Supplies default values for all other parameters required by the function.
# 3. Expose this as a callable function or FastAPI/Flask endpoint

@app.get("/generate")
async def generate(prompt: str):
    """
    Generates an image from a text prompt and returns the image as a response.
    """
    # Create a new thread for the worker if it's not already running.
    if not any(t.name == "ImageGenerationWorker" for t in threading.enumerate()):
        thread = threading.Thread(target=worker, daemon=True, name="ImageGenerationWorker")
        thread.start()
        time.sleep(5)  # Give the worker thread time to initialize

    # Create a new AsyncTask with the provided prompt and default parameters.
    task = AsyncTask(args=[])
    task.prompt = prompt
    async_tasks.append(task)

    # Wait for the task to complete and get the result.
    while not task.yields:
        await asyncio.sleep(0.1)

    image_path = None
    while True:
        if not task.yields:
            await asyncio.sleep(0.1)
            continue

        flag, product = task.yields.pop(0)
        if flag == 'finish':
            image_path = product[-1]
            break

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    else:
        return {"error": "Image generation failed"}

# Mount the FastAPI app on top of the Gradio app
gr.mount_gradio_app(shared.gradio_root, app, "/")

if __name__ == "__main__":
    # This part is for standalone testing if needed, but the primary use is mounting on Gradio.
    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=7860)
