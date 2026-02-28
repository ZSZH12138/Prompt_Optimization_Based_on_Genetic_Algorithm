import uuid
from django.http import StreamingHttpResponse, JsonResponse
from prompt_optim.pipeline import PromptOptimizerPipeline
from django.shortcuts import render

from .utils.clear_cache import clear_plot_cache
from .utils.visualize import (
    get_best_prompt_gen11,
    plot_fitness_curve,
    plot_boxplot
)


# Create your views here.
TASK_STORE = {}  # demo 用，后面可换 redis

def optimizer_page(request):
    return render(request, "optimizer/optimizer.html")

def submit_optimizer_task(request):

    clear_plot_cache()
    task_id = str(uuid.uuid4())

    TASK_STORE[task_id] = {
        "requirement": request.POST.get("requirement"),
        "text": request.POST.get("text"),
        "title": request.POST.get("title"),
        "author": request.POST.get("author"),
    }

    return JsonResponse({"task_id": task_id})


def stream_optimizer_task(request, task_id):
    params = TASK_STORE.get(task_id)

    def event_stream():
        pipeline = PromptOptimizerPipeline(**params)
        for msg in pipeline.run():
            yield f"data: {msg}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream"
    )

def result_page(request):
    best_prompt = get_best_prompt_gen11()
    fitness_img = plot_fitness_curve()
    boxplot_img = plot_boxplot()

    print(fitness_img)
    print(boxplot_img)
    return render(
        request,
        "optimizer/result.html",
        {
            "best_prompt": best_prompt,
            "fitness_img": fitness_img,
            "boxplot_img": boxplot_img,
        }
    )