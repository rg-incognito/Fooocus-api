import threading
import time
from modules.async_worker import AsyncTask, async_tasks, worker

def generate_image(prompt: str) -> str:
    """
    Generates an image from a text prompt.

    Args:
        prompt: The text prompt to generate an image from.

    Returns:
        The absolute path to the generated image.
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
        time.sleep(0.1)

    while True:
        if not task.yields:
            time.sleep(0.1)
            continue

        flag, product = task.yields.pop(0)
        if flag == 'finish':
            # The last item in the product list is the final image path.
            return product[-1]

if __name__ == '__main__':
    # Example usage:
    image_path = generate_image("a beautiful landscape")
    print(f"Image generated at: {image_path}")
