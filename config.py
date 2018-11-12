class Cfg:
    def __init__(self):
        # Root directory for data set
        self.dataroot = "E:\\Wasserstein-GAN\\dataset"

        # Root directory for cache
        self.cache = "E:\\Wasserstein-GAN\\cache"

        # Root directory for models
        self.model = "E:\\Wasserstein-GAN\\model"

        # Number of workers for data loader
        self.workers = 4

        # Batch size during training
        self.batch_size = 128

        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = 64

        # Number of channels in the training images. For color images this is 3
        self.nc = 3

        # Size of z latent vector (i.e. size of generator input)
        self.nz = 128

        # Size of feature maps in generator
        self.ngf = 64

        # Size of feature maps in discriminator
        self.ndf = 64

        # Number of training epochs
        self.num_epochs = 500

        # Number of training discriminator iterations
        self.num_dis_iter = 5

        # Learning rate for optimizers
        self.lr = 0.0005

        # Clamping lower bound
        self.lower = -0.01

        # Clamping upper bound
        self.upper = 0.01

        # Number of GPUs available. Use 0 for CPU mode.
        self.ngpu = 1

        # Lambda of gradient penalty
        self.lamb = 10

        # Use prepared data loader
        self.prepared_dataloader = False
