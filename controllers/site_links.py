import os
import re

APP_URL = os.environ.get("APP_URL", "https://smart-traffic-control-9bi4.onrender.com")
REPO_URL = "https://github.com/sierraerick46-afk/Smart-Traffic-Control"
VIDEO_URL = ""
REPORT_FILE = "docs/Final_Technical_Report_Smart_Traffic_Control.pdf"


def video_url():
    return os.environ.get("VIDEO_URL", VIDEO_URL).strip()


def video_embed(url):
    if not url:
        return None
    m = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([\w-]{11})", url)
    if m:
        return f"https://www.youtube.com/embed/{m.group(1)}"
    m = re.search(r"drive\.google\.com/file/d/([\w-]+)", url)
    if m:
        return f"https://drive.google.com/file/d/{m.group(1)}/preview"
    return None


def links(static_folder):
    url = video_url()
    report_ok = os.path.exists(os.path.join(static_folder, REPORT_FILE))
    return {"app_url": APP_URL, "repo_url": REPO_URL, "video_url": url, "video_embed": video_embed(url),
            "report_file": REPORT_FILE if report_ok else None}
