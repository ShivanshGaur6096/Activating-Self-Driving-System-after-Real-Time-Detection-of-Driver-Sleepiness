# Activating-Self-Driving-System-after-Real-Time-Detection-of-Driver-Sleepiness
I emphasis on development of this system for my major-project in my final year. This system is *development-cum-learning* project where I get familiar with various framework and libraries in Python programming. In this repository there are two individual system which further integrated into one system.
### System 1: Driver Sleepiness Detection System
In this system you are going to see the working of:
1. OpenCV (where CV stand for Computer Vision)
2. Dlib
3. imutils
4. Playsound (or you can use Pygame)
5. subprocess (compare between os.system and subprocess to call subprogram using parent program)

Note: If you want to run this system individually comment the while condition of subprocess.Popen

### System 2: Self Driving Car
To work in this system you required few thing in your system
1. [Udacity Sefl-Driving-Sim](https://github.com/udacity/self-driving-car-sim)
2. Laptop/Desktop with good specs.

There are 3 steps in development of this system
1. Data Collection
2. Model Training 
3. Testing

For above 2 points you can refer this video in [YouTube](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiWsryBps7pAhWlmOYKHTv_AbkQwqsBMAB6BAgKEAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DEaY5QiZwSP4&usg=AOvVaw0rCXk3I-mo31e-FDd6Oa0T).  
   But I have already trained model with file `model.h5`. To run this system saperately start Udacity Sim. on **Autonomous mode** and open cmd at the location of both model.h5 file and drive.py file. Run command in cmd to start python file with model.h5 file as argument.
 ``` python drive.py model.h5```

Knowlege gained during this project are:
1. PID Controller Algorithm (for Cross Track Error) with Ziegler-Nichols Technique and Twiddle Algorithm
2. dlib vs CLM Framework
3. OpenCV
4. Adam over Stochastic Gradient Decent.
5. Convolutional neural network (using PyTorch)
6. Different Neural networks, their use, different Activation functions.
6. Flask web-framework with wsgi, etc.

#### Future work
1. Upgrade the self-driving system with object detection.
2. Try to drive car in traffic and follow the traffic light in Real-World Game GTA 5.
3. Navigate car to desired location using GPS Map.


