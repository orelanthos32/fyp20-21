class EV:
    def __init__(self,size,K):
        self.layer = []
        self.kp = 0
        self.miu_x_old = 0
        self.var_x_old = 0
        self.kl = np.zeros(size)
        self.K = np.zeros(size)
        self.K[0] = K
        self.cr = np.zeros(size)
        self.miu_NS_old = np.zeros(size)
        self.var_NS_old = np.zeros(size)
        self.miu_NHS_old = np.zeros(size)
        self.var_NHS_old = np.zeros(size)
        self.node = np.empty([size, 2])
        self.BIAS2 = np.empty([size, 2])
        self.VAR = np.empty([size, 2])
        self.miumin_NS = np.empty([size, 2])
        self.miumin_NHS = np.empty([size, 2])
        self.stdmin_NS = np.empty([size, 2])
        self.stdmin_NHS = np.empty([size, 2])
        return self