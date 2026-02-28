from django.urls import path
from .views import submit_optimizer_task, stream_optimizer_task, optimizer_page, result_page

urlpatterns = [
    path("submit/", submit_optimizer_task, name="submit_optimizer_task"),
    path("stream/<str:task_id>/", stream_optimizer_task),
    path("", optimizer_page, name="optimizer_page"),
    path("results/", result_page, name="result_page"),
]