#-*- coding: utf-8 -*-

## Import Libraries
import random, collections, time
# To Calculating
import numpy as np
# To Draw a plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

## Define Functions
# Enter for Next Step
def quit_figure(event):
    if event.key == 'enter':
        plt.close(event.canvas.figure)
# Step Function
stepmaker = lambda i: 0 if i < 0 else 1
# Choose points' colors
colorchk = lambda i: 'gs' if y_data[i]==1 else 'bs'
# Compare List to stop at optimized step    
compare = lambda x, y: \
collections.Counter(x) == collections.Counter(y)
try:
## Load Dataset from ./data.txt
    xy_data = np.loadtxt('data.txt', unpack=True)
    x_data = np.transpose(np.array(xy_data[0:-1]))
    y_data = np.transpose(np.array(xy_data[-1]))
except:
## If there's no data, take exception
    x_data = np.array([[0, 0, 1.], [1, 0, 1.], [0, 1, 1.], [1, 1, 1.]])
    y_data = np.array([1, 1, 0, 1.])
## Define Global Variables
w = np.random.rand(3) % 1
learningCycle = 100

## Training Session
for i in xrange(learningCycle):
    # Count the point
    pnum = i % len(x_data)
    x, y = x_data[pnum], y_data[pnum]
    # The key point of this code
    hypothesis = np.dot(w, x)
    error = y - stepmaker(hypothesis)
    w += error * x
    # Step to get Mid hypothesis
    if i % 1 == 0:
        # Get Mid hypothesis
        mid_hypothesis = np.dot(x_data,w)
        # Print Mid hypothesis
        print "\nStep {}".format(i+1)
        print "[pNum]\t[  Dataset  ]\t[Hypothesis]\t[Result] [Real]"
        for j in xrange(len(x_data)):
            print("  p{}.\t{}\t{:-.10f}\t    {}      {:1.0f}".format(j+1,\
             x_data[j], mid_hypothesis[j], stepmaker(mid_hypothesis[j]), y_data[j]))
        # All settings to draw the plot
        point  = np.array([0, -(w[2])/(w[1])])
        normal = np.array([w[0], w[1]])
        x1_r = np.arange(-1, 2, 0.01)
        x2 = (-w[2] - normal[0]*x1_r)*1. / normal[1]
        plts = plt.plot(x1_r, x2, 'r',\
            x_data[0][0], x_data[0][1], colorchk(0),\
            x_data[1][0], x_data[1][1], colorchk(1),\
            x_data[2][0], x_data[2][1], colorchk(2),\
            x_data[3][0], x_data[3][1], colorchk(3))
        plt.gcf().canvas.set_window_title\
        ('Perceptron Learning Rule Simulator by Sun-Jin Park, InfosoftLab.')
        plt.axis([-1, 2, -1, 2])
        plt.suptitle('Perceptron Rule - Step {}, p{} turn'.format(i+1, pnum+1), fontsize=20)
        plt.grid(True)
        plt.xlabel('$x_1$', fontsize=20)
        plt.ylabel('$x_2$', fontsize=20)
        # Show W and b on plot
        plt.text(-.7, -.8, 'W = [ {:0.3f} {:0.3f} ]'.format(w[0], w[1]), style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10}, fontsize=18)
        plt.text(.7, -.8, 'b(w3) = {:0.3f}'.format(w[2]), style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10}, fontsize=18)
        # Naming points
        plt.text(x_data[0][0]+0.05, x_data[0][1]-0.08, 'p1')
        plt.text(x_data[1][0]+0.05, x_data[1][1]-0.08, 'p2')
        plt.text(x_data[2][0]+0.05, x_data[2][1]-0.08, 'p3')
        plt.text(x_data[3][0]+0.05, x_data[3][1]-0.08, 'p4')
        # Mark a point of calculated in this step
        if pnum == 0:
            plt.text(x_data[0][0]-.07, x_data[0][1]-.08, 'O', fontsize=30, color='red')
        elif pnum == 1:
            plt.text(x_data[1][0]-.07, x_data[1][1]-.08, 'O', fontsize=30, color='red')
        elif pnum == 2:
            plt.text(x_data[2][0]-.07, x_data[2][1]-.08, 'O', fontsize=30, color='red')
        else:
            plt.text(x_data[3][0]-.07, x_data[3][1]-.08, 'O', fontsize=30, color='red')
        # Guide
        plt.text(-.4, -0.98, 'Press \"Enter\" to go to the next step')
        # Add the function of 'Enter key' on plot popup
        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        # Initializing hypothesis stack to comparing 2 lists
        hStack = []
        # Stacking
        for k in xrange(len(x_data)):
            hStack.append(stepmaker(mid_hypothesis[k]))
        # Compare to recognize optimization
        if compare(hStack, y_data):
            print '\n',' ','='*31,'\n',' ','=',' '*27,'='
            print ' ','=    Optimized at Step {}!\t='.format(i+1)
            print ' ','=',' '*27,'=\n',' ','='*31
            # Save the optimized plot to Local FS
            a = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            plt.savefig('./' + a + '.png')
            plt.show()
            break
        plt.show()

# Get Total hypothesis
hypothesis = np.dot(x_data,w)
# Print Total Result
print "\n* Total Result *"
print "[pNum]\t[  Dataset  ]\t[Hypothesis]\t[Result] [Real]"
for j in xrange(len(x_data)):
     print("  p{}.\t{}\t{:-.10f}\t    {}      {:1.0f}".format(j+1,\
      x_data[j], hypothesis[j], stepmaker(hypothesis[j]), y_data[j]))
raw_input('\n\n         *** Press Enter to Shutdown ***')
