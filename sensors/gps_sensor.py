import airsim


class GpsSensor():
    """Class to provide an interface with the GPS sensor in AirSim"""
    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.current_target = [0., 0.]
        self.turn_history = []

    def position_get(self):
        """Returns float array of default car latitude and longitude"""
        lat = self.client.getGpsData("Gps", "Car1").gnss.geo_point.latitude
        long = self.client.getGpsData("Gps", "Car1").gnss.geo_point.longitude
        return [lat, long]

    def position_get_all(self, num_cars=1):
        """Returns float array of car latitude and longitude for every car"""
        coords = []
        for n in range(num_cars):
            car = "Car" + str(n + 1)
            lat = self.client.getGpsData("Gps", car).gnss.geo_point.latitude
            long = self.client.getGpsData("Gps", car).gnss.geo_point.longitude
            coords.append([lat, long])
        return coords

    def turn_push(self):
        """Pushes coordinates of when the car has two or more turn options"""
        self.turn_history.append(self.get_position())

    def turn_pop(self):
        """Pops coordinates of the car's last turn"""
        if len(self.turn_history) < 1:
            return None
        else:
            return self.turn_history.pop()

    def turn_get_last(self):
        """Returns coordinates of the last turn"""
        if len(self.turn_history) < 1:
            return None
        else:
            return self.turn_history[-1]

    def turn_reset(self):
        """Clears the turn history"""
        self.turn_history = []

    def turn_last_distance(self):
        """Gets the distance from the last turn"""
        last = self.turn_get_last()
        return self.distance(last)

    def target_get(self):
        """Returns the current coordinate target"""
        return self.current_target

    def target_set(self, coords):
        """Updates the current coordinate target"""
        if len(coords) == 2 and isinstance(coords[0], float) and \
           isinstance(coords[1], float):
            self.current_target = coords

    def target_distance(self):
        """Returns linear distance to current coordinate target"""
        cur_pos = self.position_get()
        return abs(self.current_target[0] - cur_pos[0]) + \
            abs(self.current_target[1] - cur_pos[1])

    def distance(self, coords):
        """Returns linear distance from the current position to coordinates"""
        if len(coords) == 2 and isinstance(coords[0], float) and \
           isinstance(coords[1], float):
            cur_pos = self.position_get()
            return abs(self.coords[0] - cur_pos[0]) + \
                abs(self.coords[1] - cur_pos[1])
