import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def quit_figure(event):
    if event.key == 'enter':
        plt.close(event.canvas.figure)

# Step Function
def stepmaker(i):
	if i < 0:
		return 0
	else:
		return 1

# Dataset Define from data.txt
xy_data = np.loadtxt('data.txt', unpack=True)
x_data = np.transpose(np.array(xy_data[0:-1]))
y_data = np.transpose(np.array(xy_data[-1]))

# Variable Define
w = np.random.rand(3) % 1

learningCycle = 25

# Training Session
for i in xrange(learningCycle):
    pnum = i % len(x_data)
    x, hypothesis = x_data[pnum], y_data[pnum]
    result = np.dot(w, x)
    error = hypothesis - stepmaker(result)
    w += error * x
    
    if i % 1 == 0:
        # Get Mid Result
        mid_result = np.dot(x_data,w)
        # Print Mid Result
        print "\nStep {}".format(i+1)
        print "[pNum]\t[  Dataset  ]\t[Hypothesis]\t[Result] [Real]"
        for j in xrange(len(x_data)):
            print("  p{}.\t{}\t{:-.10f}\t    {}      {:1.0f}".format(j+1, x_data[j], mid_result[j], stepmaker(mid_result[j]), y_data[j]))
        

        # To draw a line
        point  = np.array([0, -(w[2])/(w[1])])
        normal = np.array([w[0], w[1]])
        
        x1_r = np.arange(-1, 2, 0.01)
        
        x2 = (-w[2] - normal[0]*x1_r)*1. / normal[1]
        
        
        plts = plt.plot(x1_r, x2, 'r',
            x_data[0][0], x_data[0][1], 'gs',\
            x_data[1][0], x_data[1][1], 'gs',\
            x_data[2][0], x_data[2][1], 'bs',\
            x_data[3][0], x_data[3][1], 'gs')
        plt.gcf().canvas.set_window_title('Perceptron Learning Rule Simulator by Sun-Jin Park, InfosoftLab.')


        #plt.axis([-.5, 1.5, -.5, 1.5])
        plt.axis([-1, 2, -1, 2])
        
        plt.suptitle('Perceptron Rule - Step {} of {}, p{} turn'.format(i+1, learningCycle ,pnum+1), fontsize=20)

        plt.grid(True)

        plt.xlabel('$x_1$', fontsize=20)
        plt.ylabel('$x_2$', fontsize=20)


        plt.text(-.7, -.8, 'W = [ {:0.3f} {:0.3f} ]'.format(w[0], w[1]), style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10}, fontsize=18)
        plt.text(.7, -.8, 'b(w3) = {:0.3f}'.format(w[2]), style='italic',
        bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10}, fontsize=18)

        plt.text(x_data[0][0]+0.05, x_data[0][1]-0.08, 'p1')
        plt.text(x_data[1][0]+0.05, x_data[1][1]-0.08, 'p2')
        plt.text(x_data[2][0]+0.05, x_data[2][1]-0.08, 'p3')
        plt.text(x_data[3][0]+0.05, x_data[3][1]-0.08, 'p4')

        if pnum == 0:
            plt.text(x_data[0][0]-.07, x_data[0][1]-.08, 'O', fontsize=30, color='red')
        elif pnum == 1:
            plt.text(x_data[1][0]-.07, x_data[1][1]-.08, 'O', fontsize=30, color='red')
        elif pnum == 2:
            plt.text(x_data[2][0]-.07, x_data[2][1]-.08, 'O', fontsize=30, color='red')
        else:
            plt.text(x_data[3][0]-.07, x_data[3][1]-.08, 'O', fontsize=30, color='red')

        plt.text(-.4, -0.98, 'Press \"Enter\" to go to the next step')
        
        cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
        plt.show()


        #plt.draw()
        #plt.pause(1)
        #raw_input("<Hit Enter to go to Next Step>")
        #plt.close()

        # Get Result

        


# Get Total Result
result = np.dot(x_data,w)
# Print Total Result
print "\n* Total Result *"
print "[pNum]\t[  Dataset  ]\t[Hypothesis]\t[Result] [Real]"
for j in xrange(len(x_data)):
     print("  p{}.\t{}\t{:-.10f}\t    {}      {:1.0f}".format(j+1, x_data[j], result[j], stepmaker(result[j]), y_data[j]))
raw_input('Press Enter to Shutdown')