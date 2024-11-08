import numpy as np
from matplotlib import pyplot as plt

# A basic script to visualise performance over a series of heads-up games

bob = np.array([]) # add stack history here
alc = (2 * bob[0]) - bob
t = np.arange(0, alc.size, 1)

wdw = 50
alcw = np.diff(alc)
alcwf = alcw[np.where(alcw > -wdw)]
alcwf = alcw[np.where(alcw < wdw)]

plt.plot(t, alc, 'r')
plt.plot(t, bob, 'b')
plt.xlabel('Time')
plt.ylabel('Stack')


plt.figure()
plt.hist(alcw, bins=100, color='r', alpha=0.5)
plt.xlabel('Win Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Alc Wins')

print(f"Mean net: ${np.mean(alcw):.2f} ({alcw.size})")
print(f"Mean net (small): ${np.mean(alcwf):.2f} ({alcwf.size})")
print(f"Mean win: ${np.mean(alcwf[alcwf > 0]):.2f} ({alcwf[alcwf > 0].size})")
print(f"Mean loss: ${np.mean(alcwf[alcwf < 0]):.2f} ({alcwf[alcwf < 0].size})")
print(f"Stddev: ${np.std(alcw):.2f}")

print(f"Big wins: {alcw[alcw > wdw]}")
print(f"Big losses: {alcw[alcw < -wdw]}")

plt.show()
