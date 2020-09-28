#!/usr/bin/env python

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import numpy as np
import rospy

GAS_DENSITY = 2.858

class Controller(object):
    """
    Acceleration and steering controller class.
    Acceleration is controlled via PID controller.
    Steering is calculated using YawControlle.
    """

    def __init__(self, *args, **kwargs):
        
        self.vehicle_mass = kwargs['vehicle_mass']
        self.fuel_capacity = kwargs['fuel_capacity']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']
        self.accel_limit = kwargs['accel_limit']
        self.wheel_radius = kwargs['wheel_radius']
        self.wheel_base = kwargs['wheel_base']
        self.steer_ratio = kwargs['steer_ratio']
        self.max_lat_accel = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        self.yaw_controller = YawController(self.wheel_base, self.steer_ratio, 0.1, self.max_lat_accel, self.max_steer_angle)
        
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # Minimum throttle value
        mx = 0.4 # Maximum throttle value 
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        
        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.vel_lpf = LowPassFilter(tau,ts)

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        
        # Calculates acceleration or deceleration and steering angle given current and requested velocity and time.
        
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        current_vel = self.vel_lpf.filt(current_vel)
        
        # Calculate steering angle
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        
        # Calculate velocity error
        vel_error = linear_vel - current_vel

        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        # Get velocity correction from PID controller
        throttle = self.throttle_controller.step(vel_error, sample_time)
        
        brake = 0

        # Decelerate/Stop the vehicle in case of red light
        if np.isclose(linear_vel, 0.) and current_vel < 0.1:
            throttle = 0.0
            brake = 700         # N*m
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            brake = min(700, (abs(decel) * self.vehicle_mass * self.wheel_radius)) # Torque
        
        return throttle, brake, steering
