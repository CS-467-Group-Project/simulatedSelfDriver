import airsim
import time
import sensors.gps_sensor as gps

client = airsim.CarClient()
client.confirmConnection()

client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

car_state = client.getCarState()
print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))

# go forward
car_controls.throttle = 0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward")
time.sleep(1)   # let car drive a bit

# apply brakes
car_controls.brake = 1
client.setCarControls(car_controls)
print("Apply brakes")
time.sleep(3)

time.sleep(1.0)
gps.basic_gps()