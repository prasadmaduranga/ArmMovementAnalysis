import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# Create a VideoCapture object for the default webcam
cap = cv2.VideoCapture(0)

# Create a figure with two subplots: one for the Seaborn plot and one for the webcam feed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

# Create a sample Seaborn plot
tips = sns.load_dataset("tips")
sns.barplot(x="day", y="total_bill", data=tips, ax=ax1)
ax1.set_title("Seaborn Plot")

# Continuously capture frames from the webcam and show them in the second subplot
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format for Matplotlib
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show the frame in the second subplot
    ax2.imshow(frame)
    ax2.set_title("Webcam Feed")

    # Pause for a short time to allow Matplotlib to update the plot
    plt.pause(0.01)

    # Clear the second subplot to remove the previous frame
    ax2.cla()

# Release the VideoCapture object and close the plot
cap.release()
plt.close()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')
#
# def plot_init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,
#
# def plot_update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,
#
# ani = FuncAnimation(fig, plot_update, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=plot_init )
# plt.show()
