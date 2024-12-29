from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required(login_url="/admin/login/")
def index(request):
    name = "Amyr Edmar L. Francisco"
    type = "Time in"
    time = "10:00 AM"
    schedules = [
        {
            "start_time": "08:00 AM",
            "end_time": "09:00 AM",
            "title": "Automata Theory",
        },
        {
            "start_time": "09:00 AM",
            "end_time": "10:00 AM",
            "title": "Research Writing",
        },
        {
            "start_time": "10:00 AM",
            "end_time": "11:00 AM",
            "title": "Software Engineering I",
        },
    ]
    recents = [
        {
            "name": "Angelika Louise R. Labajo",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time out",
            "time": "09:30 AM",
        },
        {
            "name": "Angelo Lance O. Seraspi",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time in",
            "time": "09:00 AM",
        },
        {
            "name": "Angelika Louise R. Labajo",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time out",
            "time": "09:30 AM",
        },
        {
            "name": "Angelo Lance O. Seraspi",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time in",
            "time": "09:00 AM",
        },
        {
            "name": "Angelika Louise R. Labajo",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time out",
            "time": "09:30 AM",
        },
        {
            "name": "Angelo Lance O. Seraspi",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time in",
            "time": "09:00 AM",
        },
        {
            "name": "Angelika Louise R. Labajo",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time out",
            "time": "09:30 AM",
        },
        {
            "name": "Angelo Lance O. Seraspi",
            "image_url": "https://img.daisyui.com/images/stock/photo-1534528741775-53994a69daeb.webp",
            "type": "Time in",
            "time": "09:00 AM",
        },
    ]

    return render(
        request,
        "index.html",
        {
            "name": name,
            "type": type,
            "time": time,
            "schedules": schedules,
            "recents": recents,
        },
    )
