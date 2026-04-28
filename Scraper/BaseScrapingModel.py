
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

class BaseScraper:
    def __init__(self, headless=True, load_images=True, user_agent=''):
        options = Options()
        gecko_driver = 'driver/geckodriver' 

        if headless:
            options.add_argument("--headless")   # New way (replaces options.headless)

        options.set_preference("dom.webnotifications.enabled", False)
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.dir", "/home/abdo/Downloads")   # Linux path
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        options.set_preference("pdfjs.disabled", True)
        options.set_preference("media.volume_scale", "0.0")

        if not load_images:
            options.set_preference("permissions.default.image", 2)

        if user_agent:
            options.set_preference("general.useragent.override", user_agent)

        service = Service(executable_path=gecko_driver)
        self.driver = webdriver.Firefox(service=service, options=options)
        self.driver.set_window_size(1200, 900)

    