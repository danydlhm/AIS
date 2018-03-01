from data_distribution import DataDistribution
from generator_distribution import GeneratorDistribution
from generator_nn import GeneratorNN
from discriminator_nn import DiscriminatorNN
from numpy import reshape,ones

d_dist = DataDistribution()
d_gen = GeneratorDistribution(range=8)
nn_gen = GeneratorNN(hidden_size=4)
nn_d = DiscriminatorNN(hidden_size=4)

batch_size = 100

for i in range(10000):
    # Update Discriminator
    x1 = d_dist.sample(batch_size)
    x2 = d_gen.sample(batch_size)
    x1 = reshape(x1, [batch_size,])
    x2 = reshape(x2, [batch_size, ])

    print("Training Discriminator:")
    nn_d.train(x=x1, y=ones([batch_size, ]), batch_size=batch_size, epochs=1)
    nn_d.train(x=x2, y=ones([batch_size, ]), batch_size=batch_size, epochs=1)

    # Update Generator
    print("Training Generator:")
    x2 = d_gen.sample(batch_size)
    x2 = reshape(x2, [batch_size, ])

    nn_gen.train(x=x2, y=ones([batch_size, ]), batch_size=batch_size, epochs=1)