import time
from time import sleep
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait




chromeOptions = webdriver.ChromeOptions()
prefs = {"download.default_directory" : "Wakehurst/data"}
chromeOptions.add_experimental_option("prefs",prefs)

def scrape(urls, start_date, end_date):
	'''
	Expects

	Inputs: 
		urls - array of urls corresponding to assets
		start_date - dataset start date
		end_date - dataset end date (i.e. current date)
	'''
	driver = webdriver.Chrome(executable_path='chromedriver',options=chromeOptions)
	for i, url in enumerate(urls):
		print(f"Current URL: {url}")
		driver.get(url)
		#a GDPR accept cookies pops up the first time, we need to close it
		try:
			value = WebDriverWait(driver,3).until(EC.presence_of_element_located((By.XPATH, "//*[contains(@id, 'pop-frame')]")))
			driver.switch_to.frame(value)
			WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.call'))).click()
			driver.switch_to.default_content()
			print("Closed GDPR")
		except:
			print("Failed to close GDPR")
		#sometimes a ad pops up
		try:
			element_2 = driver.find_element_by_css_selector('i.popupCloseIcon.largeBannerCloser').click()
			print("Closed Popup")
		except:
			print("no popup yet")
		#login
		try:
			WebDriverWait(driver,3).until(EC.element_to_be_clickable((By.CLASS_NAME, 'login')))
			driver.find_element_by_class_name('login').click()
			element = driver.find_element_by_id('loginFormUser_email')
			element.send_keys('todustandtozero@gmail.com')
			element2 = driver.find_element_by_id('loginForm_password')
			element2.send_keys('pass1234',Keys.RETURN)
			print("Logged in ")
		except:
			print("Could not login")


		#set start and end date
		try:
			time.sleep(2)
			element3 = driver.find_element_by_css_selector("div.generalOverlay.js-general-overlay.displayNone")
			driver.execute_script("arguments[0].style.visibility='hidden'", element3)
			element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "widgetFieldDateRange")))
			element.click()
			# print("CLICKED")
			element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "startDate")))
			element.clear()
			element.send_keys(start_date)
			element2 = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "endDate")))
			element2.clear()
			element2.send_keys(end_date,Keys.RETURN)
			print("DOWNLOADED")
		except:
				print("Failed to set start, end date")
		#wait for the table to load and then download
		time.sleep(2)
		try:
			element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.newBtn.LightGray.downloadBlueIcon.js-download-data')))
			element.click()
		except:
			print("Failed to click download")
	driver.quit()
	return