import numpy as np
import matplotlib.pyplot as plt


def activation_Function(x, g, theta, alpha):
    return g * np.log(1 + np.exp(alpha * (x - theta))) / alpha

# Explore this activation function
fig, axs = plt.subplots(nrows=3, ncols=3, figsize = (15,15))

z = np.linspace(-2.5,2.5)
axs[0,0].plot(z,activation_Function(z,g=1,theta=1,alpha=1), linewidth = 3)
axs[0,0].set_title("g=1,theta=1,alpha=1", fontsize = 15)

axs[0,1].plot(z,activation_Function(z,g=-1,theta=1,alpha=1), linewidth = 3)
axs[0,1].set_title("g=-1,theta=1,alpha=1", fontsize = 15)

axs[0,2].plot(z,activation_Function(z,g=10,theta=1,alpha=1), linewidth = 3)
axs[0,2].set_title("g=10,theta=1,alpha=1", fontsize = 15)

axs[1,0].plot(z,activation_Function(z,g=1,theta=1,alpha=-1), linewidth = 3)
axs[1,0].set_title("g=1,theta=1,alpha=-1", fontsize = 15)

axs[1,1].plot(z,activation_Function(z,g=-1,theta=2,alpha=-1), linewidth = 3)
axs[1,1].set_title("g=-1,theta=2,alpha=-1", fontsize = 15)

axs[1,2].plot(z,activation_Function(z,g=1,theta=-1,alpha=-10), linewidth = 3)
axs[1,2].set_title("g=1,theta=-1,alpha=-10", fontsize = 15)

axs[2,0].plot(z,activation_Function(z,g=-1,theta=-1.5,alpha=-1), linewidth = 3)
axs[2,0].set_title("g=-1,theta=-1.5,alpha=-1", fontsize = 15)

axs[2,1].plot(z,activation_Function(z,g=2,theta=1.5,alpha=5), linewidth = 3)
axs[2,1].set_title("g=2,theta=1.5,alpha=5", fontsize = 15)

axs[2,2].plot(z,activation_Function(z,g=-2,theta=-1,alpha=-10), linewidth = 3)
axs[2,2].set_title("g=-2,theta=-1,alpha=-10", fontsize = 15)

axs[0, 0].spines[['right', 'top']].set_visible(False)
axs[0, 1].spines[['right', 'top']].set_visible(False)
axs[1, 0].spines[['right', 'top']].set_visible(False)
axs[1, 1].spines[['right', 'top']].set_visible(False)
axs[0, 2].spines[['right', 'top']].set_visible(False)
axs[2, 1].spines[['right', 'top']].set_visible(False)
axs[2, 0].spines[['right', 'top']].set_visible(False)
axs[2, 2].spines[['right', 'top']].set_visible(False)
axs[1, 2].spines[['right', 'top']].set_visible(False)


axs[0, 0].spines[['bottom', 'left']].set_linewidth(3)
axs[0, 1].spines[['bottom', 'left']].set_linewidth(3)
axs[1, 0].spines[['bottom', 'left']].set_linewidth(3)
axs[1, 1].spines[['bottom', 'left']].set_linewidth(3)
axs[2, 0].spines[['bottom', 'left']].set_linewidth(3)
axs[2, 1].spines[['bottom', 'left']].set_linewidth(3)
axs[2, 2].spines[['bottom', 'left']].set_linewidth(3)
axs[1, 2].spines[['bottom', 'left']].set_linewidth(3)
axs[0, 2].spines[['bottom', 'left']].set_linewidth(3)


axs[0, 0].tick_params(width=3)
axs[0, 1].tick_params(width=3)
axs[1, 0].tick_params(width=3)
axs[1, 1].tick_params(width=3)
axs[2, 2].tick_params(width=3)
axs[2, 1].tick_params(width=3)
axs[2, 0].tick_params(width=3)
axs[1, 2].tick_params(width=3)
axs[0, 2].tick_params(width=3)


axs[0, 0].set_ylim(-3,3)
axs[0, 1].set_ylim(-3,3)
axs[1, 0].set_ylim(-3,3)
axs[1, 1].set_ylim(-3,3)
axs[2, 2].set_ylim(-3,3)
axs[2, 1].set_ylim(-3,3)
axs[2, 0].set_ylim(-3,3)
axs[1, 2].set_ylim(-3,3)
axs[0, 2].set_ylim(-3,3)

axs[0, 0].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[0, 0].set_xlabel('z', fontsize = 15)
axs[0, 1].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[0, 1].set_xlabel('z', fontsize = 15)
axs[1, 0].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[1, 0].set_xlabel('z', fontsize = 15)
axs[1, 1].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[1, 1].set_xlabel('z', fontsize = 15)
axs[2, 2].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[2, 2].set_xlabel('z', fontsize = 15)
axs[2, 1].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[2, 1].set_xlabel('z', fontsize = 15)
axs[2, 0].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[2, 0].set_xlabel('z', fontsize = 15)
axs[1, 2].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[1, 2].set_xlabel('z', fontsize = 15)
axs[0, 2].set_ylabel('gamma(z;g,theta,alpha)', fontsize = 15)
axs[0, 2].set_xlabel('z', fontsize = 15)

plt.tight_layout()
plt.show()