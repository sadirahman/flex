import matplotlib.pyplot as plt;
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('school', 'work', 'student')
y_pos = np.arange(len(objects))
performance = [.857,0.923, 0.705]

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Document 5')
plt.title('WUP similarity between \n topic and label ')

fig = plt.gcf()
plt.draw()
fig.savefig('tem004.png', dpi=300)
plt.show()
