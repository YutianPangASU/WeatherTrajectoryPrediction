import csv 
import urllib.request
import json
import math

weatherFileName = "weatherfile.csv"
weatherFile = open(weatherFileName, 'a+')

latitude = "37"
longitude = "72"
timestamp = "1234556654"
altitude = "1222"

with urllib.request.urlopen("https://api.darksky.net/forecast/3843c4839a413dffd096433091e615f9/" + str(latitude) + "," + str(longitude) + "," + str(timestamp)) as url:
    data = json.loads(url.read().decode())
    try:
        pressureAtAltitude = 1013 * math.pow((1 - (2.25) * math.pow(10, -5) * float(altitude) * 0.3), 5.25)
    except:
        pressureAtAltitude = ""
    try:    
        temperatureAtAltitude = data['hourly']['data'][len(data['hourly']['data']) - 1]['temperature'] - (float(altitude) / 1000 * 3.3)
    except:
        temperatureAtAltitude = ""
    try:    
        windSpeed = data['hourly']['data'][len(data['hourly']['data']) - 1]['windSpeed']
    except:
        windSpeed = ""
    try:    
        windGust = data['hourly']['data'][len(data['hourly']['data']) - 1]['windGust']
    except:
        windGust = ""
    try:    
        windBearing = data['hourly']['data'][len(data['hourly']['data']) - 1]['windBearing']
    except:
        windBearing = ""
    try:    
        visibility = data['hourly']['data'][len(data['hourly']['data']) - 1]['visibility']
    except:
        visibility = ""
    try:    
        cloudCover = data['hourly']['data'][len(data['hourly']['data']) - 1]['cloudCover']
    except:
        cloudCover = ""
    try:    
        precipitationType = data['hourly']['data'][len(data['hourly']['data']) - 1]['precipType']
    except:    
        precipitationType = ""

    weatherData = str(pressureAtAltitude) + "," + str(temperatureAtAltitude) + "," + str(windSpeed) + "," + str(windGust) + "," + str(windBearing) + "," + str(visibility) + "," + str(cloudCover) + "," + str(precipitationType) + "," + str(altitude)
    print(weatherData)
    weatherFile.write(weatherData + "\n")
weatherFile.close()

def getTrajectory(self):
    self.cursor.execute("SELECT * FROM weather WHERE ATTRIBUTE = %s", ("" + ATTRIBUTE,))
    results = self.cursor.fetchall()
    return results



