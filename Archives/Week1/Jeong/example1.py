import numpy as np
import matplotlib.pyplot as plt

# Example 1

class RocketRailroadCar(object):
    # q: position, v: velocity
    def __init__(self, q, v):
        self.q = q
        self.v = v
        
    """ Check on which parabola the input (q,v) is and choose alpha = 1 or -1
    """
    def alpha(self, q = None, v = None):
        if q is None: q = self.q
        if v is None: v = self.v

        s = 2*q + v*abs(v) # s = 2q + v|v|
        if s == 0:
            return -1 if v > 0 else 1
        elif s > 0:
            return -1
        else:
            return 1
                
    """ Calculate the constant of the given point (q,v)
    If alpha = 1, constant = v^2 - 2q
    If alpha = -1, constant = v^2 + 2q """
    def constant(self, q = None, v = None):
        if q is None: q = self.q
        if v is None: v = self.v

        a = self.alpha(q, v)
        if a == 1:
            return v**2 - 2*q
        else:
            return v**2 + 2*q
    
    """ Find the intersection point of the parabola of (q, v) and the base curve """
    def inter(self, q = None, v = None):
        if q is None: q = self.q
        if v is None: v = self.v

        a = self.alpha(q, v)
        c = self.constant(q, v)

        if a == 1:
            q_inter = (-c)/4
            v_inter = np.sqrt(c/2)
        else:
            q_inter = c/4
            v_inter = -np.sqrt(c/2)
        
        
        return (q_inter, v_inter)
    
    def time_to_inter(self, q = None, v = None):
        if q is None: q = self.q
        if v is None: v = self.v

        # time from the given point to the base curve
        a = self.alpha(q, v)
        (q_inter, v_inter) = self.inter(q, v)
        t = (v_inter - v) * a
        return t

    def inter_to_origin(self, q = None, v = None):
        if q is None: q = self.q
        if v is None: v = self.v

        a = self.alpha(q, v)
        q_inter, v_inter = self.inter(q, v)
        t = (0 - v_inter) * (-a)
        return t


    """ Receive (q, v) and calculate the time to the origin"""
    def time_to_origin(self, q = None, v = None):
        if q is None: q = self.q
        if v is None: v = self.v

        return self.time_to_inter(q,v) + self.inter_to_origin(q,v)
    

    "Gives the velocity array corresponding to the time array"
    def velocity(self, q, v, t):
        v_array = self.alpha(q,v) * t + v
        return v_array

    "Gives the position array corresponding to the time array"
    def position(self, q, v, t):
        q_array = self.alpha(q,v)*((self.velocity(q, v, t))**2/2 - (v**2)/2) + q
        return q_array
    
    "Gives the acceleration array corresponding to the time array"
    def acceleration(self, q_array, v_array):
        a_array = np.array([])
        for i in range(len(q_array)):
            a_array = np.append(a_array, self.alpha(q_array[i], v_array[i]))
        return a_array
    
    """ Draws the wanted trajectories """
    def trajectory(self, q=None, v=None):
        if q is None: q = self.q
        if v is None: v = self.v
        
        t1 = self.time_to_inter(q, v)

        # partition [0, t1]
        assert t1 >= 0, "t is negative!"
        t1_array = np.arange(0, t1 + 0.001, 0.001)

        v_array = self.velocity(q, v, t1_array)
        q_array = self.position(q, v, t1_array)
        a_array = self.acceleration(q_array, v_array)

        t2 = self.inter_to_origin(q, v)
        q_inter, v_inter = self.inter(q,v)
        # partition [0, t2]
        assert t2 >= 0, "t is negative!"
        t2_array = np.arange(0, t2 + 0.001, 0.001)

        v_inter_array = self.velocity(q_inter, v_inter, t2_array)
        q_inter_array = self.position(q_inter, v_inter, t2_array)
        a_inter_array = self.acceleration(q_inter_array, v_inter_array)

        plt.plot(v_array, q_array)
        plt.plot(v_inter_array, q_inter_array)
        plt.xlabel("velocity")
        plt.ylabel("position")
        plt.show()

        plt.plot(q_array, t1_array)
        plt.plot(q_inter_array, t1 + t2_array)
        plt.xlabel("position")
        plt.ylabel("time")
        plt.show()

        plt.plot(v_array, t1_array)
        plt.plot(v_inter_array, t1 + t2_array)
        plt.xlabel("velocity")
        plt.ylabel("time")
        plt.show()

        plt.plot(a_array, t1_array)
        plt.plot(a_inter_array, t1 + t2_array)
        plt.xlabel("acceleration")
        plt.ylabel("time")
        plt.show()

    

        

def main():
    car1 = RocketRailroadCar(3, 2)
    print(car1.time_to_origin())
    car1.trajectory()


if __name__ == "__main__":
    main()
    
