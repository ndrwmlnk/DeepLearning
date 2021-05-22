import tensorflow as tf
import lucid.modelzoo.vision_models as models
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

def fig(imgs):
    for r in range(len(imgs)):
        for i in range(len(imgs[0])):
            plt.subplot(len(imgs), len(imgs[0]), r*len(imgs[0]) + i+1)
            plt.imshow(imgs[r][i][0])
            plt.title('step ' + str(thresholds[i]))
            plt.gca().axes.xaxis.set_visible(False)
            plt.gca().axes.yaxis.set_visible(False)

model = models.InceptionV1()

# import graph_def
with tf.Graph().as_default() as graph:
    tf.import_graph_def(model.graph_def)
# print operations
for op in graph.get_operations():
    print(op.name, op.values()[0].shape)

LEARNING_RATE = 0.05

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

# objective = "mixed4b_pre_relu:452"
# objective = "mixed3b_pre_relu:10"
objective = "mixed5b_pre_relu:1"

thresholds = (1, 32, 128, 256)  # (1, 32, 128, 256, 2048)
imgs = render.render_vis(model, objective,
                         optimizer=optimizer,
                         transforms=[],
                         param_f=lambda: param.image(64, fft=False, decorrelate=False),
                         thresholds=thresholds, verbose=True)

fig([imgs])

JITTER = 1
ROTATE = 5
SCALE = 1.1

transforms = [
    transform.pad(2*JITTER),
    transform.jitter(JITTER),
    transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
    transform.random_rotate(range(-ROTATE, ROTATE+1))
]

imgs2 = render.render_vis(model, objective, transforms=transforms,
                         param_f=lambda: param.image(64),
                         thresholds=thresholds, verbose=True)

fig([imgs, imgs2])

print('DONE')
