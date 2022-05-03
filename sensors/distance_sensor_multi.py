import airsim
import time

client = airsim.CarClient()
client.confirmConnection()

client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()

data_car1 = client.getDistanceSensorData(vehicle_name="Car1")
data_car2 = client.getDistanceSensorData(vehicle_name="Car2")
print(f"Distance sensor data: Car1: {data_car1.distance}, Car2: {data_car2.distance}")

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

data_car1 = client.getDistanceSensorData(vehicle_name="Car1")
data_car2 = client.getDistanceSensorData(vehicle_name="Car2")
print(f"Distance sensor data: Car1: {data_car1.distance}, Car2: {data_car2.distance}")

time.sleep(1.0)