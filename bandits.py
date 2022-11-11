import numpy as np
from matplotlib import pyplot as plt

class Bandit:

	def __init__(self):
		self.n = 0
		self.avg_q = 0
		self.reward = np.random.randint(100, high=200) 
		self.range = np.random.randint(10) + 1
		#print(f'''new Bandit created with reward
		#	{self.reward} and range 
		#	{self.range}''')

	def pull_arm(self):
		return (self.reward 
		+ (np.random.randint(self.range) 
		- self.range // 2))

NUM_BAND = 10
NUM_TRIALS = 100
NUM_PULLS = 10
band_arr = []
reward = [[0 for i in range(3)] 
	for i in range(NUM_TRIALS)]


def epsilon_bandit(epsilon):

	if(np.random.rand() <= epsilon):
		choice = np.random.randint(NUM_BAND)

		rew = band_arr[choice].pull_arm()

		band_arr[choice].n += 1
		band_arr[choice].avg_q += (
			(1 / band_arr[choice].n) *
			(rew - band_arr[choice].avg_q))
		#print(f"Chose bandit {choice}")
		return rew

	else:
		rew_arr = []
		for band in band_arr:
			rew_arr.append(band.avg_q)
		choice = np.argmax(rew_arr)
		rew = band_arr[choice].pull_arm()

		band_arr[choice].n += 1
		band_arr[choice].avg_q += (
			(1 / band_arr[choice].n) *
			(rew - band_arr[choice].avg_q))
		#print(f"Chose bandit {choice}")
		return rew

def UCB(c, t):
	rew_arr = []
	for band in band_arr:
		rew_arr.append(band.avg_q 
			+ c * np.sqrt(np.log(t + 1)
				/(band.n if band.n > 0 else 1)))
	choice = np.argmax(rew_arr)
	rew = band_arr[choice].pull_arm()

	band_arr[choice].n += 1
	band_arr[choice].avg_q += (
		(1 / band_arr[choice].n) *
		(rew - band_arr[choice].avg_q))
	#print(f"Chose bandit {choice}")
	return rew


def main():
	for trial in range(NUM_TRIALS):
		for i in range(NUM_BAND):
			band_arr.append(Bandit())

		for i in range(NUM_PULLS):
			reward[trial][0] += epsilon_bandit(0)
			reward[trial][1] += epsilon_bandit(0.1)
			reward[trial][2] += UCB(1, i)

	reward = np.array(reward)
	#print(reward)
	plt.plot(range(NUM_TRIALS), reward[:, 0])
	plt.plot(range(NUM_TRIALS), reward[:, 1])
	plt.plot(range(NUM_TRIALS), reward[:, 2])

	plt.show()


if __name__ == "__main__":
	main()

