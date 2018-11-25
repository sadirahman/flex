import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 5

topic_1 = (0.60, 0.66, 0.57, 0.66, 0.50)


topic_2 = (0.66, 0.75, 0.50, 0.80, 0.53)


topic_3 = (0.75, 0.72, 0.66, 0.57, 0.50)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.30

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, topic_1, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='topic 1')

rects2 = ax.bar(index + bar_width, topic_2, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='topic 2')
rects3 = ax.bar(index + (bar_width*2), topic_3, bar_width,
                alpha=opacity, color='g',
                error_kw=error_config,
                label='topic 3')

ax.set_xlabel('Documents')
ax.set_ylabel('F-measure')
ax.set_title('Scores by topics')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('document 1', 'document 2', 'document 3', 'document 4', 'document 5'))
ax.legend()

fig.tight_layout()
plt.show()