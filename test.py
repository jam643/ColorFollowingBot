#!/usr/bin/env python
import PCA9685 as servo
import time

MinPulse = 300
MaxPulse = 500
pwm = servo.PWM()
pwm.frequency = 60

if __name__ == '__main__':
    angle =  MinPulse
    for i in range(0,9):
        angle += 20
        pwm.write(15,0,angle)
        time.sleep(0.2)
    for i in range(0,9):
        angle -= 20
        pwm.write(15,0,angle)
        time.sleep(0.2)



