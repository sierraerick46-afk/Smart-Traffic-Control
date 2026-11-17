import os

APP_URL = os.environ.get("APP_URL", "https://smart-traffic-control-9bi4.onrender.com")
REPO_URL = "https://github.com/sierraerick46-afk/Smart-Traffic-Control"
REPORT_FILE = "docs/Final_Technical_Report_Smart_Traffic_Control.pdf"


def links(static_folder):
    report_ok = os.path.exists(os.path.join(static_folder, REPORT_FILE))
    return {"app_url": APP_URL, "repo_url": REPO_URL, "report_file": REPORT_FILE if report_ok else None}
