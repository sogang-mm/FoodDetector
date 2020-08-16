import matplotlib.pyplot as plt
import numpy as np
import json

train_loss = np.array(json.load(open('run-food_mobilenet-tag-loss_train_loss.json', 'r')))
valid_loss = np.array(json.load(open('run-food_mobilenet-tag-loss_valid_loss.json', 'r')))
train_prec = np.array(json.load(open('run-food_mobilenet-tag-precision_train_precision.json', 'r')))
valid_prec = np.array(json.load(open('run-food_mobilenet-tag-precision_valid_precision.json', 'r')))

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

train_loss = [train_loss[:, 1], train_loss[:, 2]]
valid_loss = [valid_loss[:, 1], valid_loss[:, 2]]

fig.suptitle('mobilenet_v2',fontsize=25)
ax1.set_title('Loss',fontsize=20)
ax1.plot(train_loss[1], label='train_loss')
ax1.plot(valid_loss[1], label='valid_loss')
ax1.legend(loc='upper left', frameon=True)
#ax1.grid()
train_prec = [train_prec[:, 1], train_prec[:, 2]]
valid_prec = [valid_prec[:, 1], valid_prec[:, 2]]

ax2.set_title('Top-1 precision',fontsize=20)
ax2.plot(train_prec[1], label='train_top1_prec')
ax2.plot(valid_prec[1], label='valid_top1_prec')
ax2.legend(loc='upper left', frameon=True)
# ax2.grid()

# fig.save('./graph.png')

plt.savefig('./graph.png',dpi=300)
